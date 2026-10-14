from pathlib import Path

from swarm import discovery, service


def test_address_preference():
    ips = ["10.145.1.104", "198.18.0.1", "8.8.8.8", "169.254.184.67"]
    assert sorted(ips, key=discovery.address_rank) == [
        "169.254.184.67", "10.145.1.104", "8.8.8.8", "198.18.0.1",
    ]


def test_resolve_order(monkeypatch):
    monkeypatch.setattr(discovery, "reachable", lambda url, timeout=1.5: True)
    monkeypatch.setattr(discovery, "find_coordinator", lambda: "http://mdns:8000")
    monkeypatch.delenv("SWARM_COORDINATOR", raising=False)
    assert discovery.resolve("http://x:1/") == "http://x:1"
    assert discovery.resolve() == "http://127.0.0.1:8000"
    assert discovery.resolve(allow_local=False) == "http://mdns:8000"
    monkeypatch.setenv("SWARM_COORDINATOR", "http://env:8000")
    assert discovery.resolve(allow_local=False) == "http://env:8000"


def test_find_coordinator_skips_unreachable_addresses(monkeypatch):
    monkeypatch.setattr(discovery, "browse", lambda timeout: [
        {"name": "m5", "addresses": ["169.254.184.67", "10.145.1.104"], "port": 8000}])
    monkeypatch.setattr(discovery, "reachable", lambda url, timeout=1.5: "10.145" in url)
    assert discovery.find_coordinator() == "http://10.145.1.104:8000"


def test_agent_plist():
    p = service.build_plist("agent", ["--model", "models/M"], Path("/proj"), "/proj/.venv/bin/python")
    assert p["Label"] == "com.mlxswarm.agent"
    assert p["ProgramArguments"] == ["/proj/.venv/bin/python", "-m", "swarm", "agent", "--model", "models/M"]
    assert p["WorkingDirectory"] == "/proj"
    assert p["KeepAlive"] == {"SuccessfulExit": False} and p["ThrottleInterval"] == 60
    assert p["StandardOutPath"].endswith("Library/Logs/mlx-swarm/agent.log")
