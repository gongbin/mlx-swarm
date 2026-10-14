"""mDNS discovery: the coordinator advertises ``_mlxswarm._tcp``; agents and the CLI find it.

A coordinator usually has several addresses (Wi-Fi, Thunderbolt bridge, VPN).
It advertises all real IPv4 ones, and the client tries them in order of
preference -- the Thunderbolt/Ethernet link-local network first, then private
LANs -- keeping the first that answers ``/health``. Proxy TUN ranges are
skipped.
"""

from __future__ import annotations

import ipaddress
import logging
import os
import socket
import time

import httpx

log = logging.getLogger("swarm.discovery")

SERVICE_TYPE = "_mlxswarm._tcp.local."
BROWSE_TIMEOUT_S = 3.0
SKIP_NETS = [ipaddress.ip_network(n) for n in ("127.0.0.0/8", "198.18.0.0/15")]  # loopback, proxy TUN


def local_ipv4s() -> list[str]:
    import ifaddr

    ips = []
    for adapter in ifaddr.get_adapters():
        for ip in adapter.ips:
            if isinstance(ip.ip, str) and not any(ipaddress.ip_address(ip.ip) in n for n in SKIP_NETS):
                ips.append(ip.ip)
    return sorted(set(ips), key=address_rank)


def address_rank(ip: str) -> tuple[int, str]:
    """Lower is better: direct cable (link-local), then private LAN, then anything else."""
    addr = ipaddress.ip_address(ip)
    if any(addr in n for n in SKIP_NETS):
        return (9, ip)
    if addr.is_link_local:
        return (0, ip)
    if addr.is_private:
        return (1, ip)
    return (2, ip)


class Advertiser:
    """Registers the coordinator on mDNS; ``close()`` withdraws it."""

    def __init__(self, port: int, name: str | None = None):
        from zeroconf import IPVersion, ServiceInfo, Zeroconf

        self.ips = local_ipv4s()
        host = socket.gethostname().split(".")[0]
        self.info = ServiceInfo(
            SERVICE_TYPE,
            f"{name or host}.{SERVICE_TYPE}",
            port=port,
            parsed_addresses=self.ips,
            properties={"api": "/v1", "pid": str(os.getpid())},
            server=f"{host}.local.",
        )
        self.zc = Zeroconf(ip_version=IPVersion.V4Only)
        self.zc.register_service(self.info, allow_name_change=True)
        log.info("advertising %s on %s port %d", self.info.name, ", ".join(self.ips), port)

    def close(self) -> None:
        try:
            self.zc.unregister_service(self.info)
        finally:
            self.zc.close()


def browse(timeout: float = BROWSE_TIMEOUT_S) -> list[dict]:
    """All coordinators answering on the local networks: [{name, addresses, port}]."""
    from zeroconf import IPVersion, ServiceBrowser, ServiceInfo, Zeroconf

    found: dict[str, dict] = {}
    zc = Zeroconf(ip_version=IPVersion.V4Only)

    class Listener:
        def add_service(self, zc, type_, name):
            info = ServiceInfo(type_, name)
            if info.request(zc, 2000):
                found[name] = {
                    "name": name.removesuffix("." + SERVICE_TYPE),
                    "addresses": sorted(
                        (ip for ip in info.parsed_addresses(IPVersion.V4Only) if address_rank(ip)[0] < 9),
                        key=address_rank,
                    ),
                    "port": info.port,
                }

        update_service = add_service

        def remove_service(self, zc, type_, name):
            found.pop(name, None)

    try:
        ServiceBrowser(zc, SERVICE_TYPE, Listener())
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            time.sleep(0.1)
            if found and time.monotonic() > deadline - timeout / 2:
                break  # got an answer and waited a little for others
    finally:
        zc.close()
    return list(found.values())


def reachable(url: str, timeout: float = 1.5) -> bool:
    try:
        return httpx.get(url + "/health", timeout=timeout).status_code == 200
    except httpx.HTTPError:
        return False


class NotFound(LookupError):
    pass


def find_coordinator(timeout: float = BROWSE_TIMEOUT_S, name: str | None = None) -> str:
    """URL of the coordinator, preferring the fastest link that actually answers."""
    services = [s for s in browse(timeout) if name is None or s["name"] == name]
    if not services:
        raise NotFound(
            "no coordinator found via mDNS (_mlxswarm._tcp); start `swarm coordinator`, "
            "or pass --coordinator / set SWARM_COORDINATOR"
        )
    if len(services) > 1:
        log.warning("several coordinators found (%s); using %s",
                    ", ".join(s["name"] for s in services), services[0]["name"])
    s = services[0]
    for ip in s["addresses"]:
        url = f"http://{ip}:{s['port']}"
        if reachable(url):
            log.info("found coordinator %s at %s", s["name"], url)
            return url
    raise NotFound(f"coordinator {s['name']} advertises {s['addresses']} but none answers /health")


def resolve(coordinator: str | None = None, allow_local: bool = True) -> str:
    """``--coordinator``, $SWARM_COORDINATOR, a coordinator on this Mac, or mDNS -- in that order.

    Agents pass ``allow_local=False``: the address they reach the coordinator
    on decides which of their own IPs they advertise, so it must not be loopback.
    """
    if coordinator:
        return coordinator.rstrip("/")
    env = os.environ.get("SWARM_COORDINATOR")
    if env:
        return env.rstrip("/")
    if allow_local and reachable("http://127.0.0.1:8000", timeout=0.5):
        return "http://127.0.0.1:8000"
    return find_coordinator()
