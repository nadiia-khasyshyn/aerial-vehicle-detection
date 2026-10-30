# -*- coding:utf8 -*-
# !/usr/bin/env python3


"""
SSH and rsync helpers for driving a RunPod pod from the local machine.
"""

import os
import shlex
import socket
import subprocess
import tempfile
import time

BASE_SSH_OPTIONS = [
    "-o", "StrictHostKeyChecking=accept-new",
    "-o", "LogLevel=ERROR",
    "-o", "ServerAliveInterval=30",
    "-o", "ServerAliveCountMax=10",
]


class RemoteError(RuntimeError):
    pass


def read_public_key(key_path: str) -> str:
    private = os.path.expanduser(key_path)
    public = f"{private}.pub"
    if not os.path.exists(private):
        raise RemoteError(f"SSH private key not found: {private}. Create one with ssh-keygen -t ed25519")
    if not os.path.exists(public):
        raise RemoteError(f"SSH public key not found: {public}. Create one with ssh-keygen -t ed25519")
    with open(public, "r") as key_file:
        return key_file.read().strip()


class RemoteShell:
    def __init__(self, host: str, port: int, user: str = "root", key_path: str = "~/.ssh/id_ed25519",
                 known_hosts_path: str | None = None):
        self.host = host
        self.port = int(port)
        self.user = user
        self.key_path = os.path.expanduser(key_path)
        if not os.path.exists(self.key_path):
            raise RemoteError(f"SSH private key not found: {self.key_path}")
        if known_hosts_path:
            self.known_hosts_path = known_hosts_path
            self.owns_known_hosts = False
        else:
            descriptor, self.known_hosts_path = tempfile.mkstemp(prefix="known-hosts-")
            os.close(descriptor)
            self.owns_known_hosts = True
        self.secrets = set()

    def close(self):
        if self.owns_known_hosts and os.path.exists(self.known_hosts_path):
            os.unlink(self.known_hosts_path)

    @property
    def ssh_options(self) -> list[str]:
        return [*BASE_SSH_OPTIONS, "-o", f"UserKnownHostsFile={self.known_hosts_path}"]

    def remember_secrets(self, values):
        self.secrets.update(value for value in values if value)

    def redact(self, text: str) -> str:
        for secret in self.secrets:
            text = text.replace(secret, "***")
        return text

    def _ssh_command(self) -> list[str]:
        return ["ssh", "-i", self.key_path, "-p", str(self.port), *self.ssh_options, f"{self.user}@{self.host}"]

    def wait_until_ready(self, timeout_seconds: int = 900, poll_interval: int = 10):
        deadline = time.time() + timeout_seconds
        while time.time() < deadline:
            if self._port_open():
                try:
                    self.run("echo ready", check=True, quiet=True)
                    return
                except RemoteError:
                    pass
            time.sleep(poll_interval)
        raise RemoteError(f"SSH on {self.host}:{self.port} did not become ready in {timeout_seconds}s")

    def _port_open(self) -> bool:
        try:
            with socket.create_connection((self.host, self.port), timeout=10):
                return True
        except OSError:
            return False

    def run(self, command: str, check: bool = True, quiet: bool = False) -> subprocess.CompletedProcess:
        full_command = [*self._ssh_command(), command]
        if not quiet:
            print(f"[remote] {self.redact(command)}")
        result = subprocess.run(full_command, capture_output=True, text=True)
        if check and result.returncode != 0:
            raise RemoteError(f"Remote command failed ({result.returncode}): {command}\n{result.stderr.strip()}")
        return result

    def stream(self, command: str) -> int:
        full_command = [*self._ssh_command(), command]
        print(f"[remote:stream] {self.redact(command)}")
        process = subprocess.Popen(full_command)
        try:
            return process.wait()
        except KeyboardInterrupt:
            process.terminate()
            raise

    def upload(self, local_path: str, remote_path: str, excludes: list[str] | None = None, delete: bool = False):
        rsync_ssh = f"ssh -i {shlex.quote(self.key_path)} -p {self.port} {' '.join(self.ssh_options)}"
        command = ["rsync", "-rlptDz", "-e", rsync_ssh]
        for pattern in excludes or []:
            command.extend(["--exclude", pattern])
        if delete:
            command.append("--delete")
        command.append(local_path)
        command.append(f"{self.user}@{self.host}:{remote_path}")
        print(f"[rsync up] {local_path} -> {remote_path}")
        result = subprocess.run(command)
        if result.returncode != 0:
            raise RemoteError(f"rsync upload failed with code {result.returncode}")

    def download_listed(self, remote_base: str, relative_paths: list[str], destination: str):
        """Fetch only the given paths, so a reused volume's history is not transferred every run."""
        if not relative_paths:
            print("No new artifacts produced by this run")
            return
        os.makedirs(destination, exist_ok=True)
        listing = tempfile.NamedTemporaryFile("w", prefix="rsync-files-", suffix=".txt", delete=False)
        try:
            listing.write("\n".join(relative_paths) + "\n")
            listing.close()
            rsync_ssh = f"ssh -i {shlex.quote(self.key_path)} -p {self.port} {' '.join(self.ssh_options)}"
            command = ["rsync", "-rlptDz", "-e", rsync_ssh,
                       f"--files-from={listing.name}",
                       f"{self.user}@{self.host}:{remote_base}/", destination]
            print(f"[rsync down] {len(relative_paths)} file(s) from {remote_base}")
            result = subprocess.run(command)
            if result.returncode != 0:
                raise RemoteError(f"rsync download failed with code {result.returncode}")
        finally:
            os.unlink(listing.name)

    def write_secret_file(self, remote_path: str, content: str, mode: str = "600"):
        """Send secrets over stdin so they never appear in a command line, a log or the pod process list."""
        command = f"umask 077; cat > {shlex.quote(remote_path)}; chmod {mode} {shlex.quote(remote_path)}"
        print(f"[remote] writing {remote_path} (contents hidden)")
        result = subprocess.run([*self._ssh_command(), command], input=content, text=True, capture_output=True)
        if result.returncode != 0:
            raise RemoteError(f"Could not write {remote_path}: {result.stderr.strip()}")

    def remove(self, remote_paths: list[str]):
        if not remote_paths:
            return
        targets = " ".join(shlex.quote(path) for path in remote_paths)
        self.run(f"rm -rf {targets}", check=False, quiet=True)

    def ssh_hint(self) -> str:
        return f"ssh -i {self.key_path} -p {self.port} {self.user}@{self.host}"


def export_prefix(environment: dict) -> str:
    parts = [f"export {key}={shlex.quote(str(value))}" for key, value in environment.items() if value is not None]
    return "; ".join(parts)
