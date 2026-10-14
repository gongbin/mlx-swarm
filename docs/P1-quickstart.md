# P1 快速上手：多台 Mac 组成一个 OpenAI 兼容推理池

本阶段是"副本池"模式（Mode R）：每台 Mac 加载一份完整模型，Coordinator 把请求分发给负载最低的节点。单机装不下的大模型仍然用 P0 的流水线脚本，P2 阶段会接入这里。

## 组件

| 命令 | 运行位置 | 作用 |
|---|---|---|
| `swarm coordinator` | 一台机器上运行一个 | OpenAI 兼容网关（`/v1/chat/completions`、`/v1/completions`、`/v1/models`），节点注册和心跳，路由与故障切换 |
| `swarm agent` | 每台提供算力的 Mac | 启动并守护 Worker，向 Coordinator 注册，每 5 秒发一次心跳（附带系统可用内存百分比） |
| `swarm worker` | 由 Agent 自动启动 | mlx-lm 的 OpenAI 服务器，加上加载前的内存/磁盘检查、内存上限，并锁定只服务启动时指定的模型 |
| `swarm nodes` / `swarm chat` | 任意位置 | 查看节点状态 / 发送一次流式请求 |

## 启动（M5 + M4 mini，直连线）

```bash
# 每台机器：同一路径下的项目，以及同名的本地模型目录
uv sync
sh bench/fetch_model.sh Qwen3-8B-4bit            # 或者用 rsync 走直连线从其他节点拷贝

# M5：Coordinator 和本机 Agent
uv run swarm coordinator --port 8000
uv run swarm agent --coordinator http://169.254.184.67:8000 --model models/Qwen3-8B-4bit --decode-concurrency 4

# mini：在「终端」App 里运行（原因见下文"本地网络权限"）
uv run swarm agent --coordinator http://169.254.184.67:8000 --model models/Qwen3-8B-4bit --decode-concurrency 4

uv run swarm nodes
uv run swarm chat "介绍一下 Apple Silicon 的统一内存" --model Qwen3-8B-4bit
```

- 客户端使用的模型名默认是模型目录名，可以用 `--served-name` 修改。
- 用于回报给 Coordinator 的 IP，Agent 会自动选择"能连到 Coordinator 的那块网卡"上的地址。Coordinator 用直连线 IP，节点就会走直连线。
- Agent 不认识的参数会原样转交给 Worker / mlx-lm，例如 `--mem-budget-gb 7`、`--decode-concurrency 8`、`--draft-model ...`。

调用示例（任何 OpenAI SDK 都可以用，`base_url` 指向 Coordinator 即可）：

```bash
curl http://169.254.184.67:8000/v1/chat/completions -H 'content-type: application/json' \
  -H 'x-session-id: user-42' \
  -d '{"model": "Qwen3-8B-4bit", "messages": [{"role": "user", "content": "hi"}], "stream": true}'
```

响应头 `x-swarm-node` 表示这个请求由哪个节点处理。

## 路由与容错

- **选节点**：在服务该模型的健康节点中，选 `(进行中的请求数 + 1) / 节点速度` 最小的那个。节点速度默认取 `results/probe-<host>.json` 里实测的内存带宽。
- **健康判断**：心跳超过 15 秒没收到、系统可用内存低于 10%、或刚失败过（冷却 10 秒）的节点，不参与分配。
- **会话亲和**：`x-session-id` 请求头，或 OpenAI 的 `user` 字段，会让同一会话固定在同一节点，以便复用 prompt cache；如果该节点比最空闲的节点多出 2 个以上进行中的请求，才换节点。
- **故障切换**：Worker 连不上，或者在响应开始前返回 5xx，就换一个节点重试，最多 2 次。已经开始的流式响应中途断开时不会重试，节点会被标记为失败。
- **Worker 崩溃**：Agent 会注销节点、重启 Worker，然后重新注册；10 分钟内崩溃超过 3 次就放弃。如果 Worker 一启动就失败（例如内存检查拒绝加载），Agent 直接退出，不会重启。

实测结果（2026-10-15，两台机器都跑 8B-4bit）：8 个并发请求两台各分到 4 个；强行杀掉 mini 的 Worker 后，6 个请求全部由 M5 成功返回；约 2 秒后 mini 自动重启并重新注册。

## 本地网络权限（macOS）

从 SSH 以 `nohup` 方式启动、并且已经脱离会话的进程，访问局域网地址会被拦截，报错为 `[Errno 65] No route to host`，而 `127.0.0.1` 不受影响。这是 macOS 的"本地网络"隐私保护。

解决方法：在节点的「终端」App 里启动 Agent；或者以 launchd 用户代理的方式运行，并在「系统设置 › 隐私与安全性 › 本地网络」里允许对应的程序。Agent 遇到这个错误时会在日志里给出提示。

## 安全

当前版本没有鉴权，Worker 和 Coordinator 都监听 `0.0.0.0`，只适合在可信的局域网或直连线上使用。共享令牌和 mTLS 放在 P5 实现。

## P1 待办

- [ ] 批量任务 API：SQLite 任务队列，支持租约、fencing、重试和去重，"有效结果只提交一次"
- [ ] mDNS 自动发现（`_mlxswarm._tcp`），不用再手动填 Coordinator 地址
- [ ] launchd 用户代理安装脚本（开机自启，并解决本地网络权限）
- [ ] `swarm top`：实时显示 tok/s、排队数和内存压力
- [ ] 更多故障注入：断网、休眠后唤醒、Coordinator 重启
