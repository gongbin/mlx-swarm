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

Without ``--coordinator`` the agent finds the coordinator via mDNS and looks
again if it stops answering (e.g. it moved to another Mac). While running, the
agent holds a power assertion so the Mac does not idle-sleep (``--allow-sleep``
to disable): a sleeping node drops its Thunderbolt link and its leases.

With ``--hostfile`` the agent runs a pipeline group instead of a single
worker: it plans the layer split from the hostfile's ``mem_budget_gb`` and
``bandwidth_gbs`` (``swarm plan``), starts the worker on every host through
``mlx.launch``, and registers rank 0 -- which must be this Mac, listed first
with ``"ssh": "127.0.0.1"`` -- as one node serving the model:

    swarm agent --model models/Qwen3-30B-A3B-6bit --hostfile hosts.json   # see hosts.example.json

Across the internet (or any NAT), ``--tunnel`` makes the agent dial out to the
coordinator over a WebSocket and receive requests through it; the worker then
listens on 127.0.0.1 only (see ``swarm.tunnel``). A coordinator with node
tokens needs one: ``--token``, ``$SWARM_NODE_TOKEN`` or the file written by
``swarm join`` (``~/.config/mlx-swarm/node-token``):

    swarm join https://swarm.example.com nt-swarm-... --model models/Qwen3-8B-4bit

Arguments the agent does not know are passed to the worker:

    swarm agent --model models/Qwen3-8B-4bit --mem-budget-gb 7 --decode-concurrency 4
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from urllib.parse import urlparse

import httpx

from swarm.discovery import NotFound, resolve
from swarm.tunnel import SCHEME as TUNNEL, TunnelClient

TOKEN_FILE = Path.home() / ".config" / "mlx-swarm" / "node-token"


def node_token(value: str | None) -> str | None:
    """--token, $SWARM_NODE_TOKEN, or the token `swarm join` saved."""
    if value or os.environ.get("SWARM_NODE_TOKEN"):
        return value or os.environ["SWARM_NODE_TOKEN"]
    try:
        return TOKEN_FILE.read_text().strip() or None
    except OSError:
        return None

log = logging.getLogger("swarm.agent")

LOAD_TIMEOUT_S = 600
MAX_RESTARTS = 3
RESTART_WINDOW_S = 600
JOB_POLL_S = 2.0
REDISCOVER_AFTER = 3  # consecutive failed heartbeats before looking for the coordinator again
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


def worker_info(worker_args: list[str]) -> dict:
    """What the dashboard shows about the worker's options."""
    def value(flag):
        return worker_args[worker_args.index(flag) + 1] if flag in worker_args[:-1] else None

    info = {}
    if draft := value("--draft-model"):
        info["draft_model"] = Path(draft.rstrip("/")).name
        info["num_draft_tokens"] = int(value("--num-draft-tokens") or 3)
    return info


class Agent:
    def __init__(self, args, worker_args: list[str]):
        self.args = args
        self.worker_args = worker_args
        self.stopping = False
        self.auto = not (args.coordinator or os.environ.get("SWARM_COORDINATOR"))
        self.set_coordinator(self.find_coordinator())
        self.local_url = f"http://127.0.0.1:{args.port}"
        self.model_name = args.served_name or Path(args.model.rstrip("/")).name
        self.node_id = args.node_id or socket.gethostname().split(".")[0]
        token = node_token(args.token)
        self.headers = {"authorization": f"Bearer {token}"} if token else {}
        self.http = httpx.Client(timeout=10.0, headers=self.headers)
        self.tunnel: TunnelClient | None = None
        self.proc: subprocess.Popen | None = None
        self.restarts: list[float] = []
        self.registered = False
        self.failed_heartbeats = 0
        self.ready = threading.Event()  # worker is serving
        self.leases: dict[int, dict] = {}  # task_id -> {"task_id", "token"}
        self.leases_lock = threading.Lock()
        self.group: dict | None = None  # pipeline group description, if any

    # -- coordinator discovery ------------------------------------------
    def find_coordinator(self) -> str:
        while True:
            try:
                return resolve(self.args.coordinator, allow_local=False)
            except NotFound as e:
                if self.stopping:
                    raise
                log.warning("%s; retrying in 10s", e)
                time.sleep(10)

    def set_coordinator(self, url: str) -> None:
        self.coordinator = url
        self.host = self.args.advertise_host or advertise_ip(url)
        # A tunnelled worker has no reachable URL; the coordinator names it tunnel://<node_id>.
        self.worker_url = TUNNEL if self.args.tunnel else f"http://{self.host}:{self.args.port}"
        if getattr(self, "tunnel", None) is not None:  # rediscovered: reconnect to the new coordinator
            self.tunnel.coordinator = url
            self.tunnel.stop_socket()
        log.info("coordinator %s; worker advertised as %s", url, self.worker_url)

    def rediscover(self) -> None:
        try:
            url = resolve(None, allow_local=False)
        except NotFound as e:
            log.warning("coordinator unreachable and %s", e)
            return
        if url != self.coordinator:
            self.set_coordinator(url)
            self.registered = False

    def keep_awake(self) -> None:
        """Prevent idle sleep for as long as this process lives."""
        try:
            subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())])
        except OSError as e:
            log.warning("could not start caffeinate: %r", e)

    # -- worker process -------------------------------------------------
    def pipeline_args(self) -> list[str]:
        """Split and per-rank budgets for a pipeline group, from the hostfile."""
        from swarm.planner import nodes_from_hostfile, plan, profile_model, report

        nodes = nodes_from_hostfile(self.args.hostfile)
        extra = []
        if "--pipeline-split" not in self.worker_args:
            profile = profile_model(self.args.model)
            p = plan(profile, nodes, self.args.plan_mode)
            log.info("pipeline plan:\n%s", report(profile, nodes, p))
            extra += ["--pipeline-split", p.split]
            self.group = {"hosts": [n.name for n in nodes], "split": p.split,
                          "expected_tok_s": round(p.expected_tok_s, 1)}
        if "--mem-budget-gb" not in self.worker_args:
            extra += ["--mem-budget-gb", ",".join(f"{n.budget_gb:g}" for n in nodes)]
        return extra

    def start_worker(self) -> None:
        worker = [
            sys.executable, "-m", "swarm.worker",
            "--model", self.args.model,
            "--host", "127.0.0.1" if self.args.tunnel else "0.0.0.0",
            "--port", str(self.args.port),
            *self.worker_args,
        ]
        if self.args.hostfile:
            launch = str(Path(sys.executable).parent / "mlx.launch")
            worker = [launch, "--backend", "ring", "--hostfile", self.args.hostfile, "--",
                      *worker, *self.pipeline_args()]
        cmd = worker
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
                    "info": {**node_info(), **worker_info(self.worker_args),
                             **({"pipeline": self.group} if self.group else {}),
                             "job_slots": self.args.job_slots},
                },
            )
            if r.status_code in (401, 403):
                log.error("coordinator refused registration: %s", r.json().get("error", {}).get("message"))
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
        if self.registered:
            try:
                r = self.http.post(
                    f"{self.coordinator}/internal/nodes/{self.node_id}/heartbeat",
                    json={"memory_free_pct": memory_free_pct(), "stats": self.stats()},
                )
                if r.status_code == 404:  # coordinator restarted and forgot us
                    self.register()
                ok = True
            except httpx.HTTPError as e:
                log.warning("heartbeat failed: %r", e)
                ok = False
        else:
            ok = self.register()
        self.failed_heartbeats = 0 if ok else self.failed_heartbeats + 1
        if self.auto and self.failed_heartbeats >= REDISCOVER_AFTER:
            self.failed_heartbeats = 0
            self.rediscover()

    def stats(self) -> dict:
        with self.leases_lock:
            leases = len(self.leases)
        return {
            "disk_free_gb": round(shutil.disk_usage(Path.home()).free / 2**30, 1),
            "load1": round(os.getloadavg()[0], 2),
            "batch_tasks": leases,
            "worker_restarts": len(self.restarts),
        }

    def deregister(self) -> None:
        try:
            self.http.post(f"{self.coordinator}/internal/nodes/{self.node_id}/deregister")
        except httpx.HTTPError:
            pass
        self.registered = False

    # -- batch tasks ----------------------------------------------------
    def job_loop(self) -> None:
        """Lease tasks while slots are free; each runs in its own thread."""
        http = httpx.Client(timeout=10.0, headers=self.headers)
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
        started = time.monotonic()
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
            report["seconds"] = round(time.monotonic() - started, 3)
            for _ in range(3):  # the coordinator may be restarting
                try:
                    resp = httpx.post(f"{self.coordinator}/internal/tasks/{task_id}/{path}",
                                      json=report, timeout=10, headers=self.headers)
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
        if not self.args.allow_sleep:
            self.keep_awake()
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
                if self.args.tunnel and self.tunnel is None:
                    self.tunnel = TunnelClient(self.coordinator, self.node_id, self.local_url, self.headers)
                    self.tunnel.start()
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
            if self.tunnel is not None:
                self.tunnel.stop()
            self.deregister()
            self.stop_worker()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="swarm agent", description=__doc__.splitlines()[0])
    parser.add_argument("--coordinator", default=None,
                        help="e.g. http://169.254.184.67:8000; default: $SWARM_COORDINATOR or mDNS")
    parser.add_argument("--model", required=True, help="local model dir or HF repo")
    parser.add_argument("--port", type=int, default=8081, help="worker port")
    parser.add_argument("--node-id", default=None, help="default: short hostname")
    parser.add_argument("--served-name", default=None, help="model name clients use; default: dir name")
    parser.add_argument("--advertise-host", default=None, help="IP the coordinator reaches the worker on")
    parser.add_argument("--speed", type=float, default=None, help="relative decode speed; default: probe")
    parser.add_argument("--heartbeat", type=float, default=5.0)
    parser.add_argument("--allow-sleep", action="store_true", help="do not keep the Mac awake")
    parser.add_argument("--hostfile", default=None,
                        help="run a pipeline group over these hosts (mlx.launch hostfile; this Mac first)")
    parser.add_argument("--plan-mode", choices=("latency", "throughput"), default="latency")
    parser.add_argument("--tunnel", action="store_true",
                        help="dial out to the coordinator and take requests over that connection "
                             "(for Macs behind NAT / on the internet); the worker listens on 127.0.0.1")
    parser.add_argument("--token", default=None,
                        help="node token from the coordinator; default: $SWARM_NODE_TOKEN or the saved one")
    parser.add_argument("--job-slots", type=int, default=2,
                        help="batch tasks run concurrently on this node (0: no batch work)")
    args, worker_args = parser.parse_known_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    sys.exit(Agent(args, worker_args).run())


if __name__ == "__main__":
    main()
