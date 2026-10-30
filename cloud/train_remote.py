"""Train on a RunPod GPU pod, or locally with --where local.

A slim port of the ml-pipeline cloud runner (train_on_runpod.py), keeping its
battle-tested pieces - pod provisioning, the detached task runner that
survives dropped SSH, crash-restart with resume, artifact download - and
dropping what this task does not have: MLproject task resolution, S3 datasets
and MLflow publishing. Code and dataset go up over rsync, results come back
into runs/cloud/<run_id>.

  python cloud/train_remote.py --list-gpus
  python cloud/train_remote.py --data dataset_final/data.yaml --gpu rtx-3090
  python cloud/train_remote.py --where local          # laptop GPU / MPS / CPU

Needs RUNPOD_API_KEY (environment or .env) and an ed25519 SSH key. Ctrl+C
detaches from the log without killing the training; reattach with
--attach-pod-id / --attach-run-id from the hint it prints.
"""

import argparse
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import yaml

from gpu_catalog import catalog_table, estimate_cost, live_catalog, resolve_gpu, suggested_batch
from provisioning import build_pod_payload, create_pod, shutdown_pod, wait_for_ssh_endpoint
from remote import RemoteError, RemoteShell, export_prefix, read_public_key
from runpod_api import RunpodClient, pod_cost_per_hour

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent / "scripts"
SECRETS_PATH = "/dev/shm/.train-env"


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    where = ap.add_argument_group("where to run")
    where.add_argument("--where", choices=["runpod", "local"], default="runpod")
    where.add_argument("--list-gpus", action="store_true")
    where.add_argument("--estimate-only", action="store_true",
                       help="print the plan and cost, create nothing")

    train = ap.add_argument_group("training (forwarded to training/train.py)")
    train.add_argument("--data", default="dataset_final/data.yaml")
    train.add_argument("--model", default="configs/vehicle_tiny_p34.yaml")
    train.add_argument("--imgsz", type=int, default=512)
    train.add_argument("--epochs", type=int, default=250)
    train.add_argument("--patience", type=int, default=60)
    train.add_argument("--batch", type=int, default=None,
                       help="default: the GPU catalog's suggestion for --imgsz")
    train.add_argument("--name", default=None, help="run name, defaults to the run id")

    pod = ap.add_argument_group("pod")
    pod.add_argument("--gpu", default="rtx-3090")
    pod.add_argument("--cloud-type", default=None, help="COMMUNITY or SECURE")
    pod.add_argument("--interruptible", action="store_true", help="spot pricing")
    pod.add_argument("--volume-gb", type=int, default=None)
    pod.add_argument("--network-volume-id", default=None)
    pod.add_argument("--max-runtime-hours", type=float, default=8.0)
    pod.add_argument("--keep-pod", action="store_true",
                     help="leave the pod running (and billing) after the task")
    pod.add_argument("--stop-instead-of-terminate", action="store_true")
    pod.add_argument("--crash-restarts", type=int, default=2)
    pod.add_argument("--follow-retries", type=int, default=20)
    pod.add_argument("--tag", default=None)
    pod.add_argument("--env-file", default=".env")
    pod.add_argument("--config", default=str(REPO_ROOT / "configs" / "runpod.yaml"))
    pod.add_argument("--code-source", choices=["rsync", "git"], default="rsync",
                     help="rsync the working tree (default) or clone repository_url")
    pod.add_argument("--git-ref", default=None, help="commit for --code-source git")
    pod.add_argument("--attach-pod-id", default=None)
    pod.add_argument("--attach-run-id", default=None)
    return ap.parse_args()


def load_env_file(path):
    values = {}
    if not os.path.exists(path):
        return values
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def run_local(args):
    """The laptop mode: training/train.py picks cuda / mps / cpu itself."""
    command = [sys.executable, str(REPO_ROOT / "training" / "train.py"),
               "--data", args.data, "--model", args.model,
               "--imgsz", str(args.imgsz), "--epochs", str(args.epochs),
               "--patience", str(args.patience)]
    if args.batch:
        command += ["--batch", str(args.batch)]
    if args.name:
        command += ["--name", args.name]
    print("[local] " + " ".join(command[1:]))
    sys.exit(subprocess.run(command).returncode)


def follow_remote_task(shell, workspace, logs_path, run_id, retries, deadline):
    """Follow the remote log, surviving dropped SSH connections without killing the task."""
    environment = (f"export WORKSPACE={shlex.quote(workspace)} LOGS_PATH={shlex.quote(logs_path)} "
                   f"RUN_ID={shlex.quote(run_id)}")
    attempt = 0
    while True:
        exit_code = shell.stream(f"{environment}; bash {shlex.quote(workspace)}/scripts/run_task.sh follow")
        if exit_code != 255:
            return exit_code
        try:
            status = shell.run(f"{environment}; bash {shlex.quote(workspace)}/scripts/run_task.sh status",
                               check=False, quiet=True).stdout.strip()
        except RemoteError:
            status = ""
        if status.startswith("finished:"):
            return int(status.split(":", 1)[1] or 1)
        attempt += 1
        if attempt > retries or time.time() > deadline:
            print(f"Lost the log stream after {attempt} attempts; the task may still be "
                  f"running on the pod, check before terminating.")
            return 255
        wait_seconds = min(60, 5 * attempt)
        print(f"SSH dropped (remote status: {status or 'unknown'}). "
              f"Reconnecting in {wait_seconds}s, attempt {attempt}/{retries}.")
        time.sleep(wait_seconds)
        try:
            shell.wait_until_ready(timeout_seconds=300, poll_interval=10)
        except RemoteError as error:
            print(f"Pod is not reachable: {error}")


def download_new_runs(shell, project_path, marker, destination):
    """Only files this run created: a reused volume keeps every previous run."""
    listing = shell.run(
        f"cd {shlex.quote(project_path)} && find runs -type f -newer {shlex.quote(marker)} "
        f"2>/dev/null || true", check=False, quiet=True)
    paths = [line.strip() for line in listing.stdout.splitlines() if line.strip()]
    try:
        shell.download_listed(project_path, paths, destination)
        return True
    except RemoteError as error:
        print(f"Could not download artifacts: {error}")
        return False


def main():
    args = parse_args()
    if args.list_gpus:
        print(catalog_table(imgsz=args.imgsz))
        return
    if args.where == "local":
        run_local(args)

    config = yaml.safe_load(open(args.config)) or {}
    gpu = resolve_gpu(args.gpu)
    cloud_type = (args.cloud_type or config.get("cloud_type", "COMMUNITY")).upper()
    gpu_count = int(config.get("gpu_count", 1))
    batch = args.batch or suggested_batch(gpu, args.imgsz, gpu_count)

    workspace = config.get("workspace_path", "/workspace")
    project_dir = config.get("project_dir_name", "aerial-vehicle-detection")
    project_path = os.path.join(workspace, project_dir)
    datasets_cache = config.get("datasets_cache_path", f"{workspace}/datasets")
    venv_path = config.get("venv_path", f"{workspace}/venv")
    logs_path = config.get("logs_path", f"{workspace}/logs")
    python_path = os.path.join(venv_path, "bin", "python")
    container_disk_gb = int(config.get("container_disk_gb", 40))
    volume_gb = args.volume_gb or int(config.get("volume_gb", 60))

    dataset_dir = (REPO_ROOT.parent / args.data).parent if not os.path.isabs(args.data) \
        else Path(args.data).parent
    # Resolve against cwd first: the dataset lives in the workspace, not the repo.
    if (Path.cwd() / args.data).exists():
        dataset_dir = (Path.cwd() / args.data).parent
    if not (dataset_dir / "data.yaml").exists():
        raise SystemExit(f"dataset not found: {args.data} (looked in {dataset_dir})")

    run_id = args.attach_run_id or \
        f"train-{datetime.now().strftime('%b-%d-%H%M')}-{args.tag or str(uuid4())[:4]}"
    run_name = args.name or run_id
    remote_data = f"datasets/{dataset_dir.name}/data.yaml"
    remote_command = (
        f"cd {shlex.quote(project_path)} && {python_path} training/train.py"
        f" --data {shlex.quote(remote_data)} --model {shlex.quote(args.model)}"
        f" --imgsz {args.imgsz} --epochs {args.epochs} --patience {args.patience}"
        f" --batch {batch} --name {shlex.quote(run_name)} --device 0")
    if args.crash_restarts > 0:
        remote_command = (f"bash {shlex.quote(workspace)}/scripts/restart_on_crash.sh "
                          f"{args.crash_restarts} {shlex.quote('--resume')} "
                          f"{shlex.quote(remote_command)}")

    live = live_catalog()
    estimate = estimate_cost(gpu, cloud_type, args.max_runtime_hours, gpu_count,
                             container_disk_gb, volume_gb, live=live)
    stock = (live.get(gpu.runpod_id) or {}).get("stock", "unknown")
    print("=" * 90)
    print(f"GPU            : {gpu.runpod_id} x{gpu_count} ({gpu.vram_gb} GB) "
          f"on {cloud_type}, stock {stock}")
    print(f"Price          : ${estimate['gpu_usd_per_hour']:.2f}/h, "
          f"ceiling ${estimate['total_usd']:.2f} for {args.max_runtime_hours}h")
    print(f"Dataset        : {dataset_dir}")
    print(f"Train          : imgsz={args.imgsz} batch={batch} epochs={args.epochs} "
          f"model={args.model}")
    print(f"Remote command : {remote_command}")
    print("=" * 90)
    if args.estimate_only:
        print("Estimate only, no pod created.")
        return

    environment = {}
    from_file = load_env_file(args.env_file)
    for key in config.get("forward_env", []):
        value = os.environ.get(key) or from_file.get(key)
        if value:
            environment[key] = value
    api_key = os.environ.get("RUNPOD_API_KEY") or from_file.get("RUNPOD_API_KEY")
    client = RunpodClient(api_key=api_key, base_url=config.get("api_base_url"))
    ssh_key_path = config.get("ssh_key_path", "~/.ssh/id_ed25519")
    public_key = read_public_key(ssh_key_path)

    follow_only = args.attach_pod_id is not None and args.attach_run_id is not None
    pod_id = args.attach_pod_id
    created_here = False
    if pod_id is None:
        payload = build_pod_payload(config, gpu, f"avd-{run_id}"[:60], public_key,
                                    cloud_type, gpu_count, container_disk_gb, volume_gb,
                                    args.network_volume_id or config.get("network_volume_id"),
                                    config.get("data_center_ids") or [],
                                    config["image_name"], args.interruptible)
        pod = create_pod(client, payload)
        pod_id = pod["id"]
        created_here = True
    else:
        print(f"Attaching to existing pod {pod_id}")

    started_at = time.time()
    deadline = started_at + args.max_runtime_hours * 3600
    exit_code, shell, detached = 1, None, False
    excludes = (config.get("secret_excludes") or []) + (config.get("upload_excludes") or [])
    try:
        pod, host, port = wait_for_ssh_endpoint(client, pod_id,
                                                int(config.get("provision_timeout_minutes", 25)),
                                                int(config.get("poll_interval_seconds", 10)))
        hourly = pod_cost_per_hour(pod) or gpu.price(cloud_type) * gpu_count
        print(f"Billing rate reported by RunPod: ${hourly:.3f}/h")

        shell = RemoteShell(host, port, config.get("ssh_user", "root"), ssh_key_path)
        shell.remember_secrets(environment.values())
        shell.wait_until_ready(timeout_seconds=int(config.get("provision_timeout_minutes", 25)) * 60)
        print(f"Pod is reachable: {shell.ssh_hint()}")

        marker = f"{logs_path}/{run_id}.marker"
        if not follow_only:
            shell.run(f"mkdir -p {shlex.quote(workspace)}/scripts {shlex.quote(logs_path)}",
                      quiet=True)
            for script in ("bootstrap.sh", "run_task.sh", "restart_on_crash.sh"):
                shell.upload(str(SCRIPTS_DIR / script), f"{workspace}/scripts/")
            shell.run(f"chmod +x {shlex.quote(workspace)}/scripts/*.sh", quiet=True)
            shell.write_secret_file(SECRETS_PATH, "".join(
                f"{key}={shlex.quote(str(value))}\n" for key, value in environment.items()))

            bootstrap_environment = {
                "WORKSPACE": workspace, "PROJECT_DIR": project_dir,
                "DATASETS_CACHE": datasets_cache, "VENV_PATH": venv_path,
                "LOGS_PATH": logs_path, "SECRETS_FILE": SECRETS_PATH,
                "REQUIREMENTS_FILE": config.get("requirements_file", "requirements.txt"),
                "CODE_SOURCE": args.code_source,
                "REPO_URL": config.get("repository_url", ""),
                "GIT_REF": args.git_ref or "",
                "INSTALL_TORCH": str(config.get("install_torch", True)).lower(),
                "TORCH_VERSION": config.get("torch_version", ""),
                "TORCHVISION_VERSION": config.get("torchvision_version", ""),
                "TORCH_INDEX_URL": config.get("torch_index_url", ""),
            }
            bootstrap = (f"{export_prefix(bootstrap_environment)}; "
                         f"bash {shlex.quote(workspace)}/scripts/bootstrap.sh")

            if args.code_source == "rsync":
                shell.run(f"mkdir -p {shlex.quote(project_path)}", quiet=True)
                shell.upload(f"{REPO_ROOT}/", f"{project_path}/", excludes=excludes,
                             delete=True)
            print("Preparing the code on the pod.")
            if shell.stream(f"{bootstrap} checkout") != 0:
                raise RuntimeError("Checkout stage failed")

            print(f"Uploading the dataset: {dataset_dir}")
            shell.run(f"mkdir -p {shlex.quote(datasets_cache)}/{shlex.quote(dataset_dir.name)}",
                      quiet=True)
            shell.upload(f"{dataset_dir}/", f"{datasets_cache}/{dataset_dir.name}/")

            print("Installing dependencies (a few minutes on a cold pod).")
            deps_id = f"{run_id}-deps"
            shell.run(f"{export_prefix({'WORKSPACE': workspace, 'LOGS_PATH': logs_path, 'RUN_ID': deps_id})}; "
                      f"bash {shlex.quote(workspace)}/scripts/run_task.sh start "
                      f"{shlex.quote(bootstrap + ' deps')}", quiet=True)
            if follow_remote_task(shell, workspace, logs_path, deps_id,
                                  args.follow_retries, deadline) != 0:
                raise RuntimeError("Dependency stage failed")

            shell.run(f"touch {shlex.quote(marker)}", quiet=True)
            sourced = f"set -a; . {shlex.quote(SECRETS_PATH)}; set +a; {remote_command}"
            shell.run(f"{export_prefix({'WORKSPACE': workspace, 'LOGS_PATH': logs_path, 'RUN_ID': run_id})}; "
                      f"bash {shlex.quote(workspace)}/scripts/run_task.sh start {shlex.quote(sourced)}",
                      quiet=True)
            print(f"Task {run_id} started. Ctrl+C detaches without killing it.")

        exit_code = follow_remote_task(shell, workspace, logs_path, run_id,
                                       args.follow_retries, deadline)
        print(f"Remote task finished with exit code {exit_code}")
        if exit_code == 255:
            detached = True
        elif exit_code == 0:
            destination = str(REPO_ROOT.parent / "runs" / "cloud" / run_id)
            if download_new_runs(shell, project_path, marker, destination):
                print(f"Artifacts downloaded to {destination}")
            else:
                detached, exit_code = True, 253
                print(f"Artifact download incomplete, pod left running: {shell.ssh_hint()}")

    except KeyboardInterrupt:
        detached = True
        print("\nDetached. The training keeps running on the pod.")
        print(f"Reattach : python cloud/train_remote.py --attach-pod-id {pod_id} "
              f"--attach-run-id {run_id}")
        print(f"Terminate: https://console.runpod.io/pods (pod {pod_id})")
    finally:
        elapsed = (time.time() - started_at) / 3600.0
        print(f"Pod uptime this session: {elapsed:.2f}h, "
              f"approximately ${gpu.price(cloud_type) * gpu_count * elapsed:.2f}")
        if shell is not None:
            shell.run(f"rm -f {shlex.quote(SECRETS_PATH)}", check=False, quiet=True)
        if (created_here or follow_only) and not args.keep_pod and not detached:
            shutdown_pod(client, pod_id, args.stop_instead_of_terminate)
        else:
            print(f"Pod {pod_id} left running and still billing.")
            if shell is not None:
                print(f"SSH: {shell.ssh_hint()}")
        if shell is not None:
            shell.close()

    if not detached:
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
