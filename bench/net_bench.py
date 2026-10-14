"""TCP latency/throughput between two Macs, shaped like pipeline traffic.

On one Mac:      python bench/net_bench.py serve [--port 5555]
On the other:    python bench/net_bench.py client <host> [--port 5555] [--label tb4]

Ping-pong sizes mirror pipeline-parallel messages:
  4 B      sampled token id
  16 KiB   one decode-step activation (hidden 8192, bf16)
  1 MiB    a prefill chunk of ~64 tokens
Throughput approximates a long-prompt prefill handoff.

Standard library only, so it runs with the system python3.
"""

import argparse
import json
import socket
import statistics
import struct
import time
from pathlib import Path

HEADER = struct.Struct("!cQQ")  # op, payload size, count
PING_SIZES = [4, 16 * 1024, 1024 * 1024]


def recv_exact(sock: socket.socket, n: int, buf: bytearray | None = None) -> bytearray:
    buf = buf if buf is not None and len(buf) >= n else bytearray(n)
    view = memoryview(buf)[:n]
    got = 0
    while got < n:
        r = sock.recv_into(view[got:], n - got)
        if r == 0:
            raise ConnectionError("peer closed")
        got += r
    return buf


def tune(sock: socket.socket):
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 8 * 1024 * 1024)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8 * 1024 * 1024)


def serve(port: int):
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", port))
    srv.listen(1)
    print(f"listening on :{port}", flush=True)
    while True:
        conn, addr = srv.accept()
        tune(conn)
        print(f"client {addr[0]} connected", flush=True)
        buf = bytearray(1)
        try:
            while True:
                op, size, count = HEADER.unpack(recv_exact(conn, HEADER.size))
                if op == b"P":
                    for _ in range(count):
                        buf = recv_exact(conn, size, buf)
                        conn.sendall(memoryview(buf)[:size])
                elif op == b"T":
                    for _ in range(count):
                        buf = recv_exact(conn, size, buf)
                    conn.sendall(b"k")
                elif op == b"Q":
                    break
        except ConnectionError:
            pass
        conn.close()
        print("client disconnected", flush=True)


def ping_pong(sock, size: int, iters: int) -> dict:
    payload = bytes(size)
    buf = bytearray(size)
    sock.sendall(HEADER.pack(b"P", size, iters + 10))
    for _ in range(10):  # warmup
        sock.sendall(payload)
        recv_exact(sock, size, buf)
    rtts = []
    for _ in range(iters):
        tic = time.perf_counter()
        sock.sendall(payload)
        recv_exact(sock, size, buf)
        rtts.append((time.perf_counter() - tic) * 1e6)
    rtts.sort()
    return {
        "size_bytes": size,
        "iters": iters,
        "rtt_us_p50": round(statistics.median(rtts), 1),
        "rtt_us_p99": round(rtts[int(len(rtts) * 0.99) - 1], 1),
        "rtt_us_min": round(rtts[0], 1),
    }


def throughput(sock, total_mb: int, chunk_mb: int = 4) -> dict:
    chunk = bytes(chunk_mb * 1024 * 1024)
    count = max(1, total_mb // chunk_mb)
    sock.sendall(HEADER.pack(b"T", len(chunk), count))
    tic = time.perf_counter()
    for _ in range(count):
        sock.sendall(chunk)
    recv_exact(sock, 1)
    elapsed = time.perf_counter() - tic
    return {
        "total_mb": count * chunk_mb,
        "seconds": round(elapsed, 3),
        "gbps": round(count * len(chunk) * 8 / elapsed / 1e9, 2),
    }


def client(host: str, port: int, label: str, iters: int, total_mb: int, out: str):
    sock = socket.create_connection((host, port), timeout=60)
    tune(sock)
    result = {
        "label": label,
        "local": socket.gethostname(),
        "peer": host,
        "ping_pong": [],
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    for size in PING_SIZES:
        n = iters if size <= 16 * 1024 else max(50, iters // 10)
        r = ping_pong(sock, size, n)
        result["ping_pong"].append(r)
        print(
            f"ping-pong {size:>8} B: p50 {r['rtt_us_p50']:>9.1f} us  "
            f"p99 {r['rtt_us_p99']:>9.1f} us",
            flush=True,
        )
    t = throughput(sock, total_mb)
    result["throughput"] = t
    print(f"throughput: {t['gbps']} Gbit/s ({t['total_mb']} MB in {t['seconds']} s)")
    sock.sendall(HEADER.pack(b"Q", 0, 0))
    sock.close()

    Path(out).mkdir(parents=True, exist_ok=True)
    name = f"net-{result['local'].split('.')[0]}-{host}-{label}.json"
    path = Path(out) / name
    path.write_text(json.dumps(result, indent=2))
    print(f"Saved to {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve")
    s.add_argument("--port", type=int, default=5555)
    c = sub.add_parser("client")
    c.add_argument("host")
    c.add_argument("--port", type=int, default=5555)
    c.add_argument("--label", default="lan", help="link type, e.g. wifi, 1gbe, 10gbe, tb4")
    c.add_argument("--iters", type=int, default=1000)
    c.add_argument("--total-mb", type=int, default=1024)
    c.add_argument("--out", default="results")
    args = parser.parse_args()

    if args.cmd == "serve":
        serve(args.port)
    else:
        client(args.host, args.port, args.label, args.iters, args.total_mb, args.out)


if __name__ == "__main__":
    main()
