"""Node agent: runs a worker on this Mac and keeps it registered with the coordinator.

The agent starts ``swarm worker`` as a child process, waits until the model is
loaded, registers the worker's URL with the coordinator and then heartbeats
every few seconds (with the Mac's free-memory percentage, so the coordinator
can stop routing to a node under memory pressure). A worker that crashes after
it was serving is restarted a limited number of times; one that fails to start
(for example, the memory guard refused the model) is not.

While the worker is up, the agent also pulls batch tasks for its model from
the coordinator (``--job-slots`` at a time, 0 to disable), runs them against
the local worker and posts the results with the lease's fencing token.

Arguments the agent does not know are passed to the worker:

    swarm agent --coordinator http://169.254.184.67:8000 --model models/Qwen3-8B-4bit \\
        --mem-budget-gb 7 --decode-concurrency 4
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from urllib.parse import urlparse

import httpx

log = logging.getLogger("swarm.agent")

LOAD_TIMEOUT_S = 600
MAX_RESTARTS = 3
RESTART_WINDOW_S = 600
JOB_POLL_S = 2.0
TASK_TIMEOUT_S = 900


def advertise_ip(coordinator_url: str) -> str:
    """The local IP the coordinator can reach us on: the one routing to it."""
    url = urlparse(coordinator_url)
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.connect((url.hostname, url.port or 80))  # UDP connect sends nothing
        return s.getsockname()[0]


def memory_free_pct() -> float | None:
    try:
        out = subprocess.run(
            ["memory_pressure", "-Q"], capture_output=True, text=True, timeout=5
        ).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    m = re.search(r"free percentage:\s*(\d+)%", out)
    return float(m.group(1)) if m else None


def node_speed(default: float = 100.0) -> float:
    """Relative decode speed: measured memory bandwidth from bench/probe.py if available."""
    host = socket.gethostname().split(".")[0]
    for path in Path("results").glob(f"probe-{host}*.json"):
        try:
            return float(json.loads(path.read_text())["measured"]["mem_bandwidth_gbs"])
        except (KeyError, ValueError, json.JSONDecodeError):
            continue
    return default


def node_info() -> dict:
    def sysctl(name):
        return subprocess.run(["sysctl", "-n", name], capture_output=True, text=True).stdout.strip()

    return {
        "hostname": socket.gethostname(),
        "chip": sysctl("machdep.cpu.brand_string"),
        "mem_total_gb": round(int(sysctl("hw.memsize") or 0) / 2**30, 1),
    }


class Agent:
    def __init__(self, args, worker_args: list[str]):
        self.args = args
        self.worker_args = worker_args
        self.coordinator = args.coordinator.rstrip("/")
        self.host = args.advertise_host or advertise_ip(self.coordinator)
        self.worker_url = f"http://{self.host}:{args.port}"
        self.local_url = f"http://127.0.0.1:{args.port}"
        self.model_name = args.served_name or Path(args.model.rstrip("/")).name
        self.node_id = args.node_id or socket.gethostname().split(".")[0]
        self.http = httpx.Client(timeout=10.0)
        self.proc: subprocess.Popen | None = None
        self.restarts: list[float] = []
        self.stopping = False
        self.registered = False
        self.ready = threading.Event()  # worker is serving
        self.leases: dict[int, dict] = {}  # task_id -> {"task_id", "token"}
        self.leases_lock = threading.Lock()

    # -- worker process -------------------------------------------------
    def start_worker(self) -> None:
        cmd = [
            sys.executable, "-m", "swarm.worker",
            "--model", self.args.model,
            "--host", "0.0.0.0",
            "--port", str(self.args.port),
            *self.worker_args,
        ]
        log.info("starting worker: %s", " ".join(cmd))
        self.proc = subprocess.Popen(cmd)

    def wait_ready(self) -> bool:
        """Wait for the HTTP server, then run a 1-token request so the model is loaded."""
        deadline = time.monotonic() + LOAD_TIMEOUT_S
        while time.monotonic() < deadline and not self.stopping:
            if self.proc.poll() is not None:
                return False
            try:
                if self.http.get(self.local_url + "/health", timeout=2).status_code == 200:
                    break
            except httpx.TransportError:
                pass
            time.sleep(1)
        else:
            return False
        try:
            r = self.http.post(
                self.local_url + "/v1/completions",
                json={"model": "default_model", "prompt": "Hi", "max_tokens": 1},
                timeout=LOAD_TIMEOUT_S,
            )
            return r.status_code == 200
        except httpx.TransportError:
            return False

    def stop_worker(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.proc.kill()

    # -- coordinator ----------------------------------------------------
    def register(self) -> bool:
        try:
            r = self.http.post(
                self.coordinator + "/internal/nodes/register",
                json={
                    "node_id": self.node_id,
                    "url": self.worker_url,
                    "model": self.model_name,
                    "speed": self.args.speed or node_speed(),
                    "info": node_info(),
                },
            )
            r.raise_for_status()
            log.info("registered %s (%s at %s)", self.node_id, self.model_name, self.worker_url)
            self.registered = True
        except httpx.HTTPError as e:
            log.warning("register failed: %r", e)
            if "No route to host" in str(e):
                log.warning(
                    "macOS may be blocking local-network access for this process; start the "
                    "agent from Terminal or allow it under Privacy & Security > Local Network"
                )
            self.registered = False
        return self.registered

    def heartbeat(self) -> None:
        if not self.registered:
            self.register()
            return
        try:
            r = self.http.post(
                f"{self.coordinator}/internal/nodes/{self.node_id}/heartbeat",
                json={"memory_free_pct": memory_free_pct()},
            )
            if r.status_code == 404:  # coordinator restarted and forgot us
                self.register()
        except httpx.HTTPError as e:
            log.warning("heartbeat failed: %r", e)

    def deregister(self) -> None:
        try:
            self.http.post(f"{self.coordinator}/internal/nodes/{self.node_id}/deregister")
        except httpx.HTTPError:
            pass
        self.registered = False

    # -- batch tasks ----------------------------------------------------
    def job_loop(self) -> None:
        """Lease tasks while slots are free; each runs in its own thread."""
        http = httpx.Client(timeout=10.0)
        next_renew = 0.0
        while not self.stopping:
            if not (self.ready.is_set() and self.registered):
                time.sleep(1)
                continue
            with self.leases_lock:
                held = list(self.leases.values())
            if held and time.monotonic() >= next_renew:
                next_renew = time.monotonic() + held[0]["lease_s"] / 3
                try:
                    r = http.post(self.coordinator + "/internal/tasks/renew",
                                  json={"node_id": self.node_id, "leases": held})
                    for task_id in r.json().get("lost", []):
                        log.warning("lost lease on task %s; its result will be discarded", task_id)
                except (httpx.HTTPError, ValueError) as e:
                    log.warning("lease renewal failed: %r", e)
            free = self.args.job_slots - len(held)
            tasks = []
            if free > 0:
                try:
                    r = http.post(self.coordinator + "/internal/tasks/lease",
                                  json={"node_id": self.node_id, "max": free})
                    tasks = r.json().get("tasks", []) if r.status_code == 200 else []
                except (httpx.HTTPError, ValueError) as e:
                    log.warning("lease failed: %r", e)
            for t in tasks:
                with self.leases_lock:
                    self.leases[t["task_id"]] = {"task_id": t["task_id"], "token": t["token"],
                                                 "lease_s": t["lease_s"]}
                threading.Thread(target=self.run_task, args=(t,), daemon=True).start()
            if not tasks:
                time.sleep(JOB_POLL_S)

    def run_task(self, task: dict) -> None:
        task_id, token = task["task_id"], task["token"]
        report = {"node_id": self.node_id, "token": token}
        try:
            try:
                r = httpx.post(self.local_url + task["url"],
                               json={**task["body"], "model": "default_model"},
                               timeout=TASK_TIMEOUT_S)
                if r.status_code == 200:
                    path, report = "complete", {**report, "result": r.json(), "model": self.model_name}
                else:
                    # 4xx is the request's fault (no point retrying); 5xx may be transient.
                    path, report = "fail", {**report, "error": f"HTTP {r.status_code}: {r.text[:500]}",
                                            "retryable": r.status_code >= 500}
            except (httpx.HTTPError, ValueError) as e:
                path, report = "fail", {**report, "error": repr(e), "retryable": True}
            for _ in range(3):  # the coordinator may be restarting
                try:
                    resp = httpx.post(f"{self.coordinator}/internal/tasks/{task_id}/{path}",
                                      json=report, timeout=10)
                    if resp.status_code == 409:
                        log.warning("task %s: %s", task_id, resp.json()["error"]["message"])
                    break
                except httpx.HTTPError as e:
                    log.warning("reporting task %s failed: %r", task_id, e)
                    time.sleep(2)
            log.info("task %s %s", task_id, "done" if path == "complete" else "failed")
        finally:
            with self.leases_lock:
                self.leases.pop(task_id, None)

    # -- main loop ------------------------------------------------------
    def run(self) -> int:
        signal.signal(signal.SIGTERM, lambda *_: setattr(self, "stopping", True))
        if self.args.job_slots > 0:
            threading.Thread(target=self.job_loop, daemon=True).start()
        try:
            while not self.stopping:
                self.start_worker()
                if not self.wait_ready():
                    if self.stopping:
                        break
                    log.error("worker did not become ready (exit code %s); not restarting",
                              self.proc.poll())
                    return 1
                self.ready.set()
                while not self.stopping and self.proc.poll() is None:
                    self.heartbeat()
                    time.sleep(self.args.heartbeat)
                self.ready.clear()
                if self.stopping:
                    break
                self.deregister()
                now = time.monotonic()
                self.restarts = [t for t in self.restarts if now - t < RESTART_WINDOW_S] + [now]
                if len(self.restarts) > MAX_RESTARTS:
                    log.error("worker crashed %d times in %ds; giving up", len(self.restarts), RESTART_WINDOW_S)
                    return 1
                log.warning("worker exited with %s; restarting", self.proc.returncode)
            return 0
        except KeyboardInterrupt:
            return 0
        finally:
            self.deregister()
            self.stop_worker()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="swarm agent", description=__doc__.splitlines()[0])
    parser.add_argument("--coordinator", required=True, help="e.g. http://169.254.184.67:8000")
    parser.add_argument("--model", required=True, help="local model dir or HF repo")
    parser.add_argument("--port", type=int, default=8081, help="worker port")
    parser.add_argument("--node-id", default=None, help="default: short hostname")
    parser.add_argument("--served-name", default=None, help="model name clients use; default: dir name")
    parser.add_argument("--advertise-host", default=None, help="IP the coordinator reaches the worker on")
    parser.add_argument("--speed", type=float, default=None, help="relative decode speed; default: probe")
    parser.add_argument("--heartbeat", type=float, default=5.0)
    parser.add_argument("--job-slots", type=int, default=2,
                        help="batch tasks run concurrently on this node (0: no batch work)")
    args, worker_args = parser.parse_known_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    sys.exit(Agent(args, worker_args).run())


if __name__ == "__main__":
    main()
