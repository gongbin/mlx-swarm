# mlx-swarm

[English](README.md) | 简体中文

把手头的几台 Apple Silicon Mac 组成一个 LLM 推理集群。mlx-swarm 在多台 Mac 上运行 [MLX](https://github.com/ml-explore/mlx) 模型，对外只提供一个 OpenAI 兼容 API。

## 特点

- **多台 Mac，一个 API**：任何 OpenAI SDK 都能直接调用这个集群。请求会分给最空闲的 Mac；同一会话固定在同一台机器上，以复用 prompt cache；某台机器出故障时，请求自动转到其他机器。
- **批量任务只提交一次结果**：把成千上万条 prompt 写成一个 JSONL 文件提交，各台 Mac 按自己的速度领取任务。即使某台 Mac 在运行中崩溃或休眠，租约、fencing token 和重试机制也能保证每个任务的结果只保存一次。
- **跑单机装不下的模型**：流水线并行把模型按层切分到多台 Mac 上，通过雷雳线或以太网传递中间结果，输出与单机逐 token 一致。规划器会根据各台 Mac 不同的内存和速度给出切分方案，并在加载模型之前预估速度。整个流水线组通过同一个 API 对外服务。
- **单路快、并发也不吃亏**：配上一个小草稿模型后，只有一个请求在跑时 Worker 用投机解码（约快 2 倍），一有其他请求进来就立即切换到批处理。正在生成的请求会在中途无缝转入批处理，不会停顿。
- **Web 控制台**：在浏览器里打开 Coordinator 的地址，就能看到每台 Mac 的健康状态、内存和实时 tok/s，和任意已部署的模型对话，提交和下载批量任务，创建或撤销 API Key 并查看每个 Key 的用量。
- **零配置**：Coordinator 通过 Bonjour（mDNS）广播自己，节点自动找到它并选择最快的链路。各组件以登录服务的方式运行，崩溃后自动重启。
- **适合日常在用的 Mac**：每次加载模型前都会检查内存和磁盘空间，并把内存用量限制在 macOS 安全线以内。节点只在提供服务期间保持唤醒。

## 当前功能

| 组件 | 功能 |
|---|---|
| `swarm coordinator` | OpenAI 兼容网关（`/v1/chat/completions`、`/v1/completions`、`/v1/models`，支持流式输出）、节点注册、路由与故障切换、批量任务队列（`/v1/jobs`，OpenAI Batch 格式）、可选的 API Key |
| 控制台（`http://<coordinator>:8000/`，或 `swarm ui`） | 节点状态与吞吐量、排空或移除节点、对话试用、批量任务、API Key 与调用示例，以及集群搭建和运维的内置文档。管理操作需要在 Coordinator 所在的 Mac 上进行，或使用 `--admin-token` |
| `swarm agent` | 每台 Mac 上运行一个：启动并守护 MLX Worker，自动发现 Coordinator，上报健康状态，执行批量任务。加上 `--hostfile` 时，把多台 Mac 组成一个流水线组，作为一个节点对外服务 |
| `swarm plan` | 试算：一个模型在一组 Mac 上如何切分，每台占多少内存，预计多少 tok/s |
| `swarm jobs` | 提交、查看、取消批量任务，下载结果 |
| `swarm service` | 把 Coordinator 或 Agent 安装为 launchd 登录服务 |
| `swarm nodes` / `chat` / `discover` | 查看集群状态、快速对话、mDNS 查找 Coordinator |
| `bench/` | 节点探测、网络微基准、单机与流水线测速 |

```bash
uv sync
uv run swarm service install coordinator                              # 在其中一台 Mac 上
uv run swarm service install agent --model models/Qwen3-8B-4bit      # 在每台 Mac 上
uv run swarm chat "你好" --model Qwen3-8B-4bit
uv run swarm jobs submit tasks.jsonl --model Qwen3-8B-4bit --wait -o results.jsonl
```

路线图：双机实测 30B MoE、流水线内多请求交错、流水线组故障切换（P2）；更多单机加速，如前缀缓存和 KV 量化（P3）。

## 性能实测

测试环境：M5 MacBook Pro（32 GB，内存带宽 122 GB/s）+ M4 Mac mini（16 GB，101 GB/s），用雷雳 3 线直连（6.5 Gbit/s，往返 0.26 ms）。

**流水线并行（单路解码速度）**

| 模型 | 单机 M5 | M5 + mini | 说明 |
|---|---|---|---|
| Qwen3-8B-4bit | 27.9 tok/s | 25.0 tok/s | 达到理论上限的 93.5%，输出与单机一致 |
| Qwen3-32B-4bit | 6.7 tok/s | 6.1 tok/s | 达到理论上限的 91%，输出与单机一致 |
| Qwen3-30B-A3B-4bit（MoE） | 59.1 tok/s | 51.4 tok/s | 输出与单机一致 |
| Qwen3-30B-A3B-6bit（MoE，23 GiB） | 装不下 | 35–37 tok/s | 只有切分到两台机器上才能运行 |

**副本池（两台机器都运行 Qwen3-8B-4bit）**

| 测试 | 结果 |
|---|---|
| 8 个并发对话请求 | 两台各分到 4 个 |
| 请求进行中杀掉一台的 Worker | 所有请求都由另一台成功返回，被杀的 Worker 约 2 秒后自动恢复 |
| 40 条 prompt 的批量任务 | 16 秒完成 |
| 批量任务进行中 `kill -9` 一台机器并重启 Coordinator | 40/40 成功，没有任务被重复执行 |
| `kill -9` 一台正持有 4 个任务的机器 | 约 30 秒后这 4 个任务被收回，在另一台完成 |
| `kill -9` 以服务方式运行的 Agent | 7 秒后被 launchd 拉起，11 秒内重新注册 |

**自适应投机解码（M5，Qwen3-8B-4bit，草稿模型 Qwen3-0.6B-4bit）**

| 负载 | 只用批处理 | 只用投机解码 | 自适应 |
|---|---|---|---|
| 每次一个请求 | 23.4 tok/s | 40.9 tok/s | 48–52 tok/s |
| 4 个并发请求（合计） | 80.2 tok/s | 40.2 tok/s | 75–81 tok/s |
| 长请求生成中途来了 3 个短请求 | — | 要等长请求结束 | 约 1 秒内全部返回 |

## 许可证

[Apache License 2.0](LICENSE)
