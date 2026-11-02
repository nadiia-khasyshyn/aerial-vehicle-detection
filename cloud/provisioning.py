"""
Pod provisioning: builds the RunPod create-pod payload and waits until the machine is reachable.
"""

import time

from gpu_catalog import GpuOption
from runpod_api import RunpodClient, RunpodError, pod_status, resolve_ssh_endpoint


def build_pod_payload(config: dict, gpu: GpuOption, pod_name: str, public_key: str,
                      cloud_type: str, gpu_count: int,
                      container_disk_gb: int, volume_gb: int, network_volume_id: str | None,
                      data_center_ids: list[str], image_name: str, interruptible: bool) -> dict:
    gpu_priority = [gpu.runpod_id]
    for fallback in config.get("gpu_priority", []):
        if fallback not in gpu_priority:
            gpu_priority.append(fallback)

    pod_environment = {"PUBLIC_KEY": public_key}

    payload = {
        "name": pod_name[:191],
        "imageName": image_name,
        "computeType": "GPU",
        "cloudType": cloud_type,
        "gpuTypeIds": gpu_priority,
        "gpuTypePriority": "custom",
        "gpuCount": gpu_count,
        "containerDiskInGb": container_disk_gb,
        "volumeInGb": volume_gb,
        "volumeMountPath": config.get("volume_mount_path", "/workspace"),
        "ports": config.get("ports", ["22/tcp"]),
        "env": pod_environment,
        "interruptible": interruptible,
        "supportPublicIp": bool(config.get("support_public_ip", True)),
    }

    if config.get("allowed_cuda_versions"):
        payload["allowedCudaVersions"] = config["allowed_cuda_versions"]
    if config.get("min_ram_per_gpu_gb"):
        payload["minRAMPerGPU"] = int(config["min_ram_per_gpu_gb"])
    if config.get("min_vcpu_per_gpu"):
        payload["minVCPUPerGPU"] = int(config["min_vcpu_per_gpu"])
    if config.get("min_download_mbps"):
        payload["minDownloadMbps"] = int(config["min_download_mbps"])
    if config.get("min_upload_mbps"):
        payload["minUploadMbps"] = int(config["min_upload_mbps"])
    if data_center_ids:
        payload["dataCenterIds"] = data_center_ids
        payload["dataCenterPriority"] = "custom"
    if network_volume_id:
        payload["networkVolumeId"] = network_volume_id
        payload.pop("volumeInGb", None)

    return payload


def create_pod(client: RunpodClient, payload: dict) -> dict:
    print(f"Creating pod '{payload['name']}' on {payload['cloudType']} with {payload['gpuTypeIds'][0]}")
    pod = client.create_pod(payload)
    pod_id = pod.get("id")
    if not pod_id:
        raise RunpodError(f"RunPod did not return a pod id: {pod}")
    print(f"Pod created: {pod_id}")
    return pod


def wait_for_ssh_endpoint(client: RunpodClient, pod_id: str, timeout_minutes: int,
                          poll_interval: int) -> tuple[dict, str, int]:
    deadline = time.time() + timeout_minutes * 60
    last_status = None
    while time.time() < deadline:
        pod = client.get_pod(pod_id)
        status = pod_status(pod)
        if status != last_status:
            print(f"Pod {pod_id} status: {status}")
            last_status = status
        if status in ("TERMINATED", "FAILED", "EXITED"):
            raise RunpodError(f"Pod {pod_id} reached terminal status {status} before becoming reachable")
        endpoint = resolve_ssh_endpoint(pod)
        if status == "RUNNING" and endpoint is not None:
            host, port = endpoint
            print(f"Pod {pod_id} SSH endpoint: {host}:{port}")
            return pod, host, port
        time.sleep(poll_interval)
    raise RunpodError(
        f"Pod {pod_id} did not expose SSH within {timeout_minutes} minutes. "
        f"Check the RunPod console, the pod is still billing."
    )


def shutdown_pod(client: RunpodClient, pod_id: str, stop_instead_of_terminate: bool, attempts: int = 5):
    action = client.stop_pod if stop_instead_of_terminate else client.terminate_pod
    verb = "stopped. The volume keeps billing at the idle rate" if stop_instead_of_terminate else "terminated"
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            action(pod_id)
            print(f"Pod {pod_id} {verb}.")
            return
        except Exception as error:
            last_error = error
            if attempt < attempts:
                time.sleep(min(30, 2 ** attempt))
    print(f"Failed to shut down pod {pod_id} after {attempts} attempts: {last_error}")
    print("It is still billing. Terminate it manually: https://console.runpod.io/pods")
