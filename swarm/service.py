"""Run the coordinator or an agent as a launchd login agent (starts at login, restarts on crash).

    swarm service install agent --model models/Qwen3-8B-4bit --decode-concurrency 4
    swarm service install coordinator --port 8000
    swarm service status [agent|coordinator]
    swarm service logs agent [-n 50]
    swarm service uninstall agent

Arguments after the role are stored in the plist and passed to ``swarm <role>``
as is. The service runs this project's Python from this directory, so relative
model paths keep working. Logs go to ~/Library/Logs/mlx-swarm/<role>.log.

The first time a launchd-started process reaches the local network, macOS asks
for permission (System Settings > Privacy & Security > Local Network); until it
is allowed, LAN connections fail with "No route to host".

macOS also guards ~/Documents, ~/Desktop and ~/Downloads: a launchd process
reading them blocks until access is granted (Privacy & Security > Files and
Folders, or Full Disk Access for the Python binary). Keeping the project
elsewhere, e.g. ~/mlx-swarm, avoids that.
"""

from __future__ import annotations

import argparse
import os
import plistlib
import subprocess
import sys
import time
from pathlib import Path

ROLES = ("agent", "coordinator")
LAUNCH_AGENTS = Path.home() / "Library" / "LaunchAgents"
LOG_DIR = Path.home() / "Library" / "Logs" / "mlx-swarm"
# Minimum seconds between launches (also delays `launchctl kickstart -k`).
THROTTLE_S = {"agent": 60, "coordinator": 10}
PROTECTED_DIRS = [Path.home() / d for d in ("Documents", "Desktop", "Downloads")]


def label(role: str) -> str:
    return f"com.mlxswarm.{role}"


def plist_path(role: str) -> Path:
    return LAUNCH_AGENTS / f"{label(role)}.plist"


def log_path(role: str) -> Path:
    return LOG_DIR / f"{role}.log"


def domain() -> str:
    return f"gui/{os.getuid()}"


def build_plist(role: str, args: list[str], workdir: Path, python: str) -> dict:
    return {
        "Label": label(role),
        "ProgramArguments": [python, "-m", "swarm", role, *args],
        "WorkingDirectory": str(workdir),
        "RunAtLoad": True,
        "KeepAlive": {"SuccessfulExit": False},  # restart after crashes, not after a clean stop
        "ThrottleInterval": THROTTLE_S[role],
        "ProcessType": "Interactive",  # no background CPU/IO throttling
        "StandardOutPath": str(log_path(role)),
        "StandardErrorPath": str(log_path(role)),
        "EnvironmentVariables": {
            "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",  # caffeinate, memory_pressure, sysctl
            "PYTHONUNBUFFERED": "1",
            **({"SWARM_COORDINATOR": os.environ["SWARM_COORDINATOR"]}
               if "SWARM_COORDINATOR" in os.environ else {}),
        },
    }


def launchctl(*args: str, check: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(["launchctl", *args], capture_output=True, text=True, check=check)


def unload(role: str, timeout: float = 60.0) -> None:
    """``launchctl bootout`` returns before the job is gone; wait until it is."""
    target = f"{domain()}/{label(role)}"
    if launchctl("print", target).returncode != 0:
        return
    launchctl("bootout", target)
    deadline = time.monotonic() + timeout
    while launchctl("print", target).returncode == 0 and time.monotonic() < deadline:
        time.sleep(0.5)


def install(role: str, args: list[str]) -> None:
    workdir = Path.cwd().resolve()
    if not (workdir / "swarm" / "__init__.py").exists():
        sys.exit("run `swarm service install` from the mlx-swarm project directory")
    if role == "agent" and "--model" not in args:
        sys.exit("an agent needs --model, e.g. swarm service install agent --model models/Qwen3-8B-4bit")
    LAUNCH_AGENTS.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = plist_path(role)
    unload(role)  # replace a previous install
    with open(path, "wb") as f:
        plistlib.dump(build_plist(role, args, workdir, sys.executable), f)
    r = launchctl("bootstrap", domain(), str(path))
    if r.returncode != 0:
        sys.exit(f"launchctl bootstrap failed: {r.stderr.strip() or r.stdout.strip()}")
    print(f"installed {label(role)} -> {path}")
    print(f"logs: {log_path(role)}")
    print("If macOS asks whether Python may find devices on the local network, click Allow "
          "(System Settings > Privacy & Security > Local Network).")
    protected = next((d for d in PROTECTED_DIRS if workdir.is_relative_to(d)), None)
    if protected:
        print(f"\nWARNING: the project is inside ~/{protected.name}, which macOS protects. The service "
              f"will hang at startup until {Path(sys.executable).resolve()} is allowed to access it "
              f"(Privacy & Security > Files and Folders / Full Disk Access). Moving the project to "
              f"e.g. ~/mlx-swarm avoids this.")


def uninstall(role: str) -> None:
    unload(role)
    path = plist_path(role)
    if path.exists():
        path.unlink()
        print(f"removed {path}")
    else:
        print(f"{label(role)} was not installed")


def status(roles: list[str]) -> None:
    for role in roles:
        if not plist_path(role).exists():
            print(f"{role}: not installed")
            continue
        r = launchctl("print", f"{domain()}/{label(role)}")
        if r.returncode != 0:
            print(f"{role}: installed but not loaded (swarm service install {role} ... to reload)")
            continue
        fields = {}
        for line in r.stdout.splitlines():
            key, sep, value = line.strip().partition(" = ")
            if sep and key in ("state", "pid", "last exit code", "runs"):
                fields.setdefault(key, value)
        args = plistlib.loads(plist_path(role).read_bytes())["ProgramArguments"][3:]
        print(f"{role}: {fields.get('state', '?')}  pid {fields.get('pid', '-')}  "
              f"runs {fields.get('runs', '?')}  last exit {fields.get('last exit code', '-')}")
        print(f"  args: {' '.join(args)}")
        print(f"  logs: {log_path(role)}")


def logs(role: str, lines: int) -> None:
    path = log_path(role)
    if not path.exists():
        sys.exit(f"no log yet at {path}")
    subprocess.run(["tail", "-n", str(lines), str(path)])


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[2:] if argv is None else argv)
    if not argv or argv[0] not in ("install", "uninstall", "status", "logs"):
        sys.exit(__doc__)
    cmd, rest = argv[0], argv[1:]
    if cmd == "status":
        status([r for r in rest if r in ROLES] or list(ROLES))
        return
    if not rest or rest[0] not in ROLES:
        sys.exit(f"usage: swarm service {cmd} {{{'|'.join(ROLES)}}} ...")
    role, rest = rest[0], rest[1:]
    if cmd == "install":
        install(role, rest)
    elif cmd == "uninstall":
        uninstall(role)
    else:
        parser = argparse.ArgumentParser(prog="swarm service logs")
        parser.add_argument("-n", type=int, default=50)
        logs(role, parser.parse_args(rest).n)
