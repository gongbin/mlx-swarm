"""Probe this Mac's capabilities and write a node profile JSON.

Usage:
    python bench/probe.py [--out results/] [--quick]

Measures GPU memory bandwidth (the bound on decode tok/s) and fp16 matmul
throughput (the bound on prefill tok/s), and records hardware/OS/network/RDMA
facts the planner needs.
"""

import argparse
import json
import platform
import re
import shutil
import socket
import subprocess
import time
from importlib import metadata
from pathlib import Path

import mlx.core as mx


def sh(cmd: str) -> str:
    try:
        return subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=30
        ).stdout.strip()
    except subprocess.TimeoutExpired:
        return ""


def sysctl(name: str) -> str:
    return sh(f"sysctl -n {name}")


def gpu_cores() -> int | None:
    m = re.search(r'"gpu-core-count"\s*=\s*(\d+)', sh("ioreg -rc AGXAccelerator -d1"))
    return int(m.group(1)) if m else None


def thunderbolt_speeds() -> list[str]:
    out = sh("system_profiler SPThunderboltDataType")
    return sorted(set(re.findall(r"Speed:\s*(.+)", out)))


def network_interfaces() -> list[dict]:
    """IPv4 interfaces that are up, with their hardware port names."""
    ports = {}
    for block in sh("networksetup -listallhardwareports").split("\n\n"):
        name = re.search(r"Hardware Port:\s*(.+)", block)
        dev = re.search(r"Device:\s*(\S+)", block)
        if name and dev:
            ports[dev.group(1)] = name.group(1)

    result = []
    for dev in sh("ifconfig -l").split():
        ip = sh(f"ipconfig getifaddr {dev}")
        if ip and not dev.startswith("lo"):
            result.append({"device": dev, "port": ports.get(dev, ""), "ipv4": ip})
    return result


def rdma_status() -> dict:
    if not shutil.which("ibv_devices"):
        return {"available": False, "devices": []}
    devices = [
        line.split()[0]
        for line in sh("ibv_devices").splitlines()[2:]
        if line.strip()
    ]
    return {"available": bool(devices), "devices": devices}


def measure_bandwidth(size_gb: float, iters: int) -> float:
    """Read-dominated GPU bandwidth in GB/s (reduction over a large fp16 array)."""
    n = int(size_gb * 1e9 / 2)
    a = mx.random.normal((n,)).astype(mx.float16)
    mx.eval(a)
    for _ in range(2):
        mx.eval(mx.sum(a))
    tic = time.perf_counter()
    for _ in range(iters):
        mx.eval(mx.sum(a))
    elapsed = time.perf_counter() - tic
    del a
    mx.clear_cache()
    return n * 2 * iters / elapsed / 1e9


def measure_tflops(dim: int, iters: int) -> float:
    a = mx.random.normal((dim, dim)).astype(mx.float16)
    b = mx.random.normal((dim, dim)).astype(mx.float16)
    mx.eval(a, b)
    for _ in range(2):
        mx.eval(a @ b)
    tic = time.perf_counter()
    for _ in range(iters):
        mx.eval(a @ b)
    elapsed = time.perf_counter() - tic
    return 2 * dim**3 * iters / elapsed / 1e12


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="results")
    parser.add_argument("--quick", action="store_true", help="smaller, faster tests")
    args = parser.parse_args()

    mem_total = int(sysctl("hw.memsize"))
    wired_limit_mb = sysctl("iogpu.wired_limit_mb")
    info = mx.device_info()

    print("Measuring GPU memory bandwidth ...", flush=True)
    bw = measure_bandwidth(0.5 if args.quick else 2.0, 5 if args.quick else 20)
    print("Measuring fp16 matmul throughput ...", flush=True)
    tflops = measure_tflops(4096, 5 if args.quick else 20)

    profile = {
        "hostname": socket.gethostname(),
        "chip": sysctl("machdep.cpu.brand_string"),
        "model_id": sysctl("hw.model"),
        "cpu_cores": {
            "performance": int(sysctl("hw.perflevel0.physicalcpu") or 0),
            "efficiency": int(sysctl("hw.perflevel1.physicalcpu") or 0),
        },
        "gpu_cores": gpu_cores(),
        "mem_total_gb": round(mem_total / 2**30, 1),
        "gpu_wired_limit_mb": int(wired_limit_mb) if wired_limit_mb.isdigit() else None,
        "mlx_max_recommended_working_set_gb": round(
            info.get("max_recommended_working_set_size", 0) / 2**30, 1
        ),
        "os": platform.mac_ver()[0],
        "python": platform.python_version(),
        "mlx": mx.__version__,
        "mlx_lm": metadata.version("mlx-lm"),
        "distributed_backends": [
            b for b in ("ring", "jaccl", "mpi") if mx.distributed.is_available(b)
        ],
        "rdma": rdma_status(),
        "thunderbolt_speeds": thunderbolt_speeds(),
        "network": network_interfaces(),
        "measured": {
            "mem_bandwidth_gbs": round(bw, 1),
            "fp16_matmul_tflops": round(tflops, 2),
        },
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"probe-{profile['hostname'].split('.')[0]}.json"
    path.write_text(json.dumps(profile, indent=2, ensure_ascii=False))
    print(json.dumps(profile, indent=2, ensure_ascii=False))
    print(f"\nSaved to {path}")


if __name__ == "__main__":
    main()
