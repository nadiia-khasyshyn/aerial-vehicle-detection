# -*- coding:utf8 -*-
# !/usr/bin/env python3


"""
RunPod GPU catalog. Prices are read from the RunPod API and fall back to the snapshot below when the
API is unreachable, batch hints are estimates to start from.
"""

import os
from dataclasses import dataclass

PRICES_UPDATED_AT = "2026-09-10"
PRICING_URL = "https://www.runpod.io/pricing"
GRAPHQL_URL = "https://api.runpod.io/graphql"
LIVE_QUERY = """query {
  gpuTypes {
    id
    communityPrice
    securePrice
    lowestPrice(input: {gpuCount: 1}) { stockStatus }
  }
}"""


@dataclass(frozen=True)
class GpuOption:
    alias: str
    runpod_id: str
    vram_gb: int
    community_usd_per_hour: float
    secure_usd_per_hour: float
    batch_hint: dict
    note: str

    def price(self, cloud_type: str) -> float:
        return self.community_usd_per_hour if cloud_type.upper() == "COMMUNITY" else self.secure_usd_per_hour


GPU_CATALOG: dict[str, GpuOption] = {
    "rtx-3090": GpuOption(
        alias="rtx-3090",
        runpod_id="NVIDIA GeForce RTX 3090",
        vram_gb=24,
        community_usd_per_hour=0.22,
        secure_usd_per_hour=0.50,
        batch_hint={256: 256, 320: 192, 416: 128, 640: 64},
        note="Ampere sm_86, 936 GB/s, FP16 142 TFLOPS. Best price per hour for 256px tile training.",
    ),
    "rtx-3070": GpuOption(
        alias="rtx-3070",
        runpod_id="NVIDIA GeForce RTX 3070",
        vram_gb=8,
        community_usd_per_hour=0.13,
        secure_usd_per_hour=0.0,
        batch_hint={256: 96, 320: 64, 416: 48, 640: 24},
        note="Ampere sm_86, 8 GB: cheapest card that fits the tiny model; batches stay small.",
    ),
    "rtx-4000-sff": GpuOption(
        alias="rtx-4000-sff",
        runpod_id="NVIDIA RTX 4000 SFF Ada Generation",
        vram_gb=20,
        community_usd_per_hour=0.18,
        secure_usd_per_hour=0.44,
        batch_hint={256: 192, 320: 128, 416: 96, 640: 48},
        note="Ada sm_89, 20 GB at 70 W: slow but roomy and very cheap per hour.",
    ),
    "rtx-4090": GpuOption(
        alias="rtx-4090",
        runpod_id="NVIDIA GeForce RTX 4090",
        vram_gb=24,
        community_usd_per_hour=0.34,
        secure_usd_per_hour=0.74,
        batch_hint={256: 256, 320: 192, 416: 128, 640: 64},
        note="Ada sm_89, roughly 1.6-1.9x faster than 3090 on the same tile size.",
    ),
    "rtx-a5000": GpuOption(
        alias="rtx-a5000",
        runpod_id="NVIDIA RTX A5000",
        vram_gb=24,
        community_usd_per_hour=0.16,
        secure_usd_per_hour=0.27,
        batch_hint={256: 192, 320: 160, 416: 96, 640: 48},
        note="Cheapest 24 GB option, slower than 3090, good for long low-priority runs.",
    ),
    "rtx-a6000": GpuOption(
        alias="rtx-a6000",
        runpod_id="NVIDIA RTX A6000",
        vram_gb=48,
        community_usd_per_hour=0.33,
        secure_usd_per_hour=0.49,
        batch_hint={256: 512, 320: 384, 416: 256, 640: 128},
        note="Ampere 48 GB, 3090-class speed with double the memory.",
    ),
    "a40": GpuOption(
        alias="a40",
        runpod_id="NVIDIA A40",
        vram_gb=48,
        community_usd_per_hour=0.35,
        secure_usd_per_hour=0.49,
        batch_hint={256: 512, 320: 384, 416: 256, 640: 128},
        note="48 GB, fits large batches of 256px tiles without gradient accumulation.",
    ),
    "l40s": GpuOption(
        alias="l40s",
        runpod_id="NVIDIA L40S",
        vram_gb=48,
        community_usd_per_hour=0.79,
        secure_usd_per_hour=1.09,
        batch_hint={256: 512, 320: 384, 416: 256, 640: 128},
        note="Ada 48 GB, fastest option below A100 for detection training.",
    ),
    "a100-pcie": GpuOption(
        alias="a100-pcie",
        runpod_id="NVIDIA A100 80GB PCIe",
        vram_gb=80,
        community_usd_per_hour=1.19,
        secure_usd_per_hour=1.59,
        batch_hint={256: 768, 320: 576, 416: 384, 640: 192},
        note="Only worth it for multi-GPU or 640px runs, overkill for 256px tiles.",
    ),
    "a100-sxm": GpuOption(
        alias="a100-sxm",
        runpod_id="NVIDIA A100-SXM4-80GB",
        vram_gb=80,
        community_usd_per_hour=1.39,
        secure_usd_per_hour=1.59,
        batch_hint={256: 768, 320: 576, 416: 384, 640: 192},
        note="SXM interconnect, relevant only for gpu_count > 1.",
    ),
}

STORAGE_USD_PER_GB_MONTH = {
    "container_disk": 0.10,
    "volume_running": 0.10,
    "volume_idle": 0.20,
    "network_volume_under_1tb": 0.07,
    "network_volume_over_1tb": 0.05,
}


def resolve_gpu(alias_or_id: str) -> GpuOption:
    key = alias_or_id.strip().lower()
    if key in GPU_CATALOG:
        return GPU_CATALOG[key]
    for option in GPU_CATALOG.values():
        if option.runpod_id.lower() == key:
            return option
    known = ", ".join(GPU_CATALOG.keys())
    raise ValueError(f"Unknown GPU '{alias_or_id}'. Known aliases: {known}")


def suggested_batch(option: GpuOption, imgsz: int, gpu_count: int = 1) -> int:
    supported = sorted(option.batch_hint.keys())
    closest = min(supported, key=lambda size: abs(size - imgsz))
    return option.batch_hint[closest] * max(gpu_count, 1)


def estimate_cost(option: GpuOption, cloud_type: str, hours: float, gpu_count: int = 1,
                  container_disk_gb: int = 0, volume_gb: int = 0, live: dict | None = None) -> dict:
    hourly = current_price(option, cloud_type, live)
    gpu_cost = hourly * max(gpu_count, 1) * hours
    disk_cost = (container_disk_gb + volume_gb) * STORAGE_USD_PER_GB_MONTH["volume_running"] / 730.0 * hours
    return {
        "gpu_usd_per_hour": hourly * max(gpu_count, 1),
        "gpu_usd": round(gpu_cost, 2),
        "storage_usd": round(disk_cost, 2),
        "total_usd": round(gpu_cost + disk_cost, 2),
    }


def live_catalog(api_key: str | None = None, timeout: int = 15) -> dict:
    """Current prices and stock per RunPod gpu id. Empty when there is no key or the API is unreachable."""
    key = api_key or os.environ.get("RUNPOD_API_KEY")
    if not key:
        return {}
    try:
        import requests
        response = requests.post(f"{GRAPHQL_URL}?api_key={key}", json={"query": LIVE_QUERY}, timeout=timeout)
        response.raise_for_status()
        types = response.json()["data"]["gpuTypes"]
    except Exception as error:
        print(f"Live GPU prices unavailable, falling back to the {PRICES_UPDATED_AT} snapshot: {str(error)[:120]}")
        return {}
    return {
        entry["id"]: {
            "community": entry.get("communityPrice"),
            "secure": entry.get("securePrice"),
            "stock": (entry.get("lowestPrice") or {}).get("stockStatus") or "none",
        }
        for entry in types
    }


def current_price(option: GpuOption, cloud_type: str, live: dict | None = None) -> float:
    entry = (live or {}).get(option.runpod_id)
    if entry:
        price = entry["community"] if cloud_type.upper() == "COMMUNITY" else entry["secure"]
        if price:
            return float(price)
    return option.price(cloud_type)


def catalog_table(cloud_type: str = "COMMUNITY", imgsz: int = 256, live: dict | None = None) -> str:
    live = live_catalog() if live is None else live
    header = (f"{'alias':<12} {'runpod gpu id':<28} {'vram':>5} {'usd/h':>7} {'stock':>8} "
              f"{'batch@' + str(imgsz):>11}  note")
    lines = [header, "-" * len(header)]
    for option in GPU_CATALOG.values():
        stock = (live.get(option.runpod_id) or {}).get("stock", "unknown")
        lines.append(
            f"{option.alias:<12} {option.runpod_id:<28} {str(option.vram_gb) + 'G':>5} "
            f"{current_price(option, cloud_type, live):>7.2f} {stock:>8} "
            f"{suggested_batch(option, imgsz):>11}  {option.note}"
        )
    lines.append("")
    source = "live from the RunPod API" if live else f"snapshot {PRICES_UPDATED_AT}, source {PRICING_URL}"
    lines.append(f"Prices: {cloud_type}, {source}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(catalog_table())
