# -*- coding:utf8 -*-
# !/usr/bin/env python3


"""
Thin client for the RunPod REST API v1 (https://rest.runpod.io/v1).
"""

import os
import time
import requests

DEFAULT_BASE_URL = "https://rest.runpod.io/v1"


class RunpodError(RuntimeError):
    pass


class RunpodClient:
    def __init__(self, api_key: str | None = None, base_url: str = DEFAULT_BASE_URL, timeout: int = 60):
        self.api_key = api_key or os.environ.get("RUNPOD_API_KEY")
        if not self.api_key:
            raise RunpodError("RUNPOD_API_KEY is not set. Put it into .env or export it before running.")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        })

    def _request(self, method: str, path: str, **kwargs):
        url = f"{self.base_url}/{path.lstrip('/')}"
        response = self.session.request(method, url, timeout=self.timeout, **kwargs)
        if response.status_code >= 400:
            raise RunpodError(f"{method} {url} failed with {response.status_code}: {response.text}")
        if not response.content:
            return {}
        try:
            return response.json()
        except ValueError:
            return {"raw": response.text}

    def create_pod(self, payload: dict) -> dict:
        return self._request("POST", "/pods", json=payload)

    def get_pod(self, pod_id: str) -> dict:
        return self._request("GET", f"/pods/{pod_id}")

    def stop_pod(self, pod_id: str) -> dict:
        return self._request("POST", f"/pods/{pod_id}/stop")

    def terminate_pod(self, pod_id: str) -> dict:
        return self._request("DELETE", f"/pods/{pod_id}")

def pod_status(pod: dict) -> str:
    return str(pod.get("desiredStatus") or pod.get("status") or "UNKNOWN").upper()


def pod_cost_per_hour(pod: dict) -> float:
    for key in ("costPerHr", "costPerHour", "cost_per_hr"):
        value = pod.get(key)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return 0.0


def resolve_ssh_endpoint(pod: dict) -> tuple[str, int] | None:
    """Return (host, port) for a direct SSH connection to the pod, or None when it is not exposed yet."""
    public_ip = pod.get("publicIp") or pod.get("ip")
    port_mappings = pod.get("portMappings") or {}
    if public_ip and isinstance(port_mappings, dict):
        mapped = port_mappings.get("22") or port_mappings.get(22)
        if mapped:
            return str(public_ip), int(mapped)

    runtime = pod.get("runtime") or {}
    for port in runtime.get("ports") or []:
        if int(port.get("privatePort", 0)) == 22 and port.get("isIpPublic"):
            return str(port.get("ip")), int(port.get("publicPort"))
    return None
