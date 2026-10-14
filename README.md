# mlx-swarm

基于 [MLX](https://github.com/ml-explore/mlx) 的多台 Apple Silicon Mac 分布式 LLM 推理：

- **副本池**：每台 Mac 加载一份完整模型，对外提供一个 OpenAI 兼容 API，带路由、会话亲和、故障切换和批量任务队列。主要用来提升吞吐。
- **流水线并行**：把模型按层切到多台 Mac 上，用来跑单机装不下的大模型（目前是测速脚本，P2 阶段会接入服务端）。

设计方案见 `doc2-技术评估与开发方案.md`。

## 安装

每台 Mac 都要装，并且项目放在**相同路径**下（流水线模式要求各机路径一致）：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # 已安装可跳过
uv sync    # 锁定 mlx 0.32.1 / mlx-lm 0.31.3；pyproject.toml 里已配置清华 PyPI 镜像
uv run swarm   # 列出所有命令
```

注意：
- 系统自带的 `python3` 是 3.9，请始终用 `uv run` 或 `.venv/bin/python` 运行。
- 主控机需要能免密 SSH 到各节点：`ssh-copy-id <user>@<node>`。

### 下载模型

国内访问 Hugging Face（包括 hf-mirror）大文件只有几十 KB/s。建议从 ModelScope 的 mlx-community 镜像下载到 `models/`，速度约 4–10 MB/s：

```bash
sh bench/fetch_model.sh Qwen3-8B-4bit              # 默认 org 为 mlx-community
sh bench/fetch_model.sh Qwen3-30B-A3B-4bit org/Name   # 可一次下载多个
```

已经下载过的模型，可以用 rsync 走直连线拷到其他节点（雷雳线上拷 17 GB 约 1 分钟）：

```bash
rsync -a models/Qwen3-8B-4bit/ <user>@<节点IP>:<项目路径>/models/Qwen3-8B-4bit/
```

## 推理池（OpenAI 兼容 API）

| 命令 | 运行位置 | 作用 |
|---|---|---|
| `swarm coordinator` | 任选一台，只运行一个 | API 网关：`/v1/chat/completions`、`/v1/completions`、`/v1/models`、`/v1/jobs`；负责节点注册、路由、故障切换和批量任务队列 |
| `swarm agent` | 每台提供算力的 Mac | 启动并守护 Worker，向 Coordinator 注册并发送心跳，同时领取批量任务 |
| `swarm worker` | 由 Agent 自动启动 | mlx-lm 的 OpenAI 服务器，加了内存/磁盘检查和内存上限，并锁定只服务启动时指定的模型 |
| `swarm nodes` / `swarm chat` | 任意位置 | 查看节点状态 / 发一次流式请求 |
| `swarm jobs` | 任意位置 | 提交和管理批量任务 |

### 启动

```bash
# Coordinator 所在机器
uv run swarm coordinator --port 8000 --db jobs.sqlite3

# 每台节点（包括 Coordinator 所在机器），各自使用本地的同名模型目录
uv run swarm agent --coordinator http://<coordinator-ip>:8000 \
    --model models/Qwen3-8B-4bit --decode-concurrency 4 --job-slots 2

uv run swarm nodes
uv run swarm chat "介绍一下 Apple Silicon 的统一内存" --model Qwen3-8B-4bit
```

- **模型名**：客户端使用的模型名默认是模型目录名（如 `Qwen3-8B-4bit`），可以用 `--served-name` 修改。
- **上报地址**：Agent 会自动选择"能连到 Coordinator 的那块网卡"的 IP 作为自己的地址。Coordinator 地址写直连线 IP，节点间流量就走直连线。
- **透传参数**：Agent 不认识的参数会原样转给 Worker / mlx-lm，例如 `--mem-budget-gb 7`、`--decode-concurrency 8`、`--draft-model ...`。
- **环境变量**：设置 `SWARM_COORDINATOR=http://<coordinator-ip>:8000` 后，`swarm nodes/chat/jobs` 就不用再写 `--coordinator`。

### 调用

任何 OpenAI SDK 都可以用，把 `base_url` 设为 `http://<coordinator-ip>:8000/v1` 即可：

```bash
curl http://<coordinator-ip>:8000/v1/chat/completions -H 'content-type: application/json' \
  -H 'x-session-id: user-42' \
  -d '{"model": "Qwen3-8B-4bit", "messages": [{"role": "user", "content": "hi"}], "stream": true}'
```

响应头 `x-swarm-node` 表示这个请求由哪个节点处理。

### 路由与容错

- **选节点**：在服务该模型的健康节点中，选 `(进行中的请求数 + 1) / 节点速度` 最小的。节点速度默认取 `bench/probe.py` 实测的内存带宽（`results/probe-<host>.json`），也可以用 `--speed` 指定。
- **健康判断**：以下节点暂不分配请求：
  - 超过 15 秒没有心跳；
  - 系统可用内存低于 10%；
  - 刚失败过，还在 10 秒冷却期内。
- **会话亲和**：`x-session-id` 请求头或 OpenAI 的 `user` 字段，会让同一会话固定在同一节点，以复用 prompt cache。只有当该节点比最空闲的节点多出 2 个以上进行中的请求时才换节点。
- **故障切换**：Worker 连不上，或者在开始响应之前返回 5xx，就换一个节点重试，最多 2 次。流式响应已经开始后中途断开不会重试，该节点会被标记为失败。
- **Worker 崩溃**：Agent 会注销节点、重启 Worker，然后重新注册；10 分钟内崩溃超过 3 次就放弃。如果 Worker 一启动就失败（例如内存检查拒绝加载模型），Agent 直接退出，不会重启。

### 批量任务

适合大量互相独立的请求，比如摘要、翻译、打标签、跑评测集。任务进入 Coordinator 的 SQLite 队列，各节点的 Agent 按自己加载的模型主动领取，跑完再交回结果。

```bash
uv run swarm jobs submit tasks.jsonl --model Qwen3-8B-4bit --key run-001 --wait -o out.jsonl
uv run swarm jobs list
uv run swarm jobs status <job-id>
uv run swarm jobs results <job-id> -o out.jsonl
uv run swarm jobs cancel <job-id>
```

输入文件每行一个任务，可以用 OpenAI Batch 格式，也可以用简写：

```jsonl
{"custom_id": "a1", "url": "/v1/chat/completions", "body": {"messages": [{"role": "user", "content": "..."}], "max_tokens": 128}}
{"custom_id": "a2", "prompt": "用一句话解释熵", "max_tokens": 64}
```

输出也是 JSONL，按提交顺序排列，每行包含 `custom_id`、`status`、`response.body`（与 OpenAI 的返回格式相同）、`error`、`node` 和 `attempts`。

HTTP 接口：

| 接口 | 说明 |
|---|---|
| `POST /v1/jobs` | 提交任务，参数为 `tasks`、`model`、`max_attempts`、`idempotency_key`、`metadata` |
| `GET /v1/jobs` | 任务列表 |
| `GET /v1/jobs/{id}` | 任务状态 |
| `GET /v1/jobs/{id}/results` | 结果 |
| `POST /v1/jobs/{id}/cancel` | 取消任务 |

**队列的语义**

- **租约与 fencing**：
  - 每次领取任务都会让该任务的 token 加 1。租约有效期 30 秒，Agent 每 10 秒续约一次。
  - 只接受持有当前 token 的结果。如果节点失联，租约到期后任务会重新排队；失联节点之后再交回的结果会被拒绝（HTTP 409）。
  - 所以每个任务的有效结果只会提交一次。
- **重试**：
  - 5xx、连接失败或租约过期：等待 2、4、8……秒后重试（最长 60 秒），直到用完 `max_attempts` 次（默认 3 次）。
  - 4xx：说明请求本身有问题，直接判定失败。
- **去重**：相同的 `idempotency_key` 只会创建一个任务，重复提交会返回原来的任务。同一个任务内 `custom_id` 不能重复。
- **持久化**：队列保存在 `--db` 指定的文件里。Coordinator 重启后任务不会丢失，Agent 手上的租约仍然有效。
- **并发**：每个节点同时运行 `--job-slots` 个批量任务（设为 0 表示不领取），和交互请求共用同一个 Worker。

### macOS 本地网络权限

从 SSH 用 `nohup` 启动、已经脱离会话的进程，访问局域网地址时会被 macOS 的"本地网络"隐私保护拦截，报错 `[Errno 65] No route to host`；访问 `127.0.0.1` 不受影响。

解决方法：
- 在节点的「终端」App 里启动 Agent；
- 或者以 launchd 用户代理运行，并在「系统设置 › 隐私与安全性 › 本地网络」中授权。

Agent 遇到这个错误时会在日志里给出提示。

## 流水线并行（测速脚本）

```bash
uv run python bench/probe.py                      # 节点探测：芯片、内存、实测带宽/算力、网卡
uv run python bench/net_bench.py serve            # 网络微基准：在节点 B 上运行
uv run python bench/net_bench.py client <B的IP> --label tb4   # 在节点 A 上运行

# 单机基线，并保存 greedy 参考输出，供流水线校验正确性
uv run python bench/llm_bench.py --model models/Qwen3-8B-4bit --save-ref --label M5

# 多机流水线（hostfile 中的顺序就是流水线顺序，主控机写成 "ssh": "127.0.0.1"）
uv run mlx.launch --backend ring --hostfile hosts-cable.json -- \
  $PWD/.venv/bin/python bench/llm_bench.py --model models/Qwen3-8B-4bit --split w:15,7 --mem-budget-gb 20,8.5
```

- `--split` 有三种写法：`even` 平均分；`10,18` 直接给每个 rank 的层数；`w:20,12` 按权重比例分（建议填各节点可用显存的 GB 数）。
- 输出中的 `correctness.identical` 必须为 `true`。
- 多机模式下，各节点的项目路径和模型路径必须一致；要显式指定解释器 `$PWD/.venv/bin/python`。
- 直连：两台 Mac 用雷雳线相连后会出现"Thunderbolt 网桥"，自动获得 169.254.x.x 地址。`hosts-cable.json` 和 `hosts-wifi.json` 是两种链路的 hostfile 示例。

### 防死机

开发初期曾两次整机卡死，原因和对策如下：

- **磁盘满导致 swap 无法扩展**：加载模型前会检查剩余磁盘空间，不足 `--min-free-disk-gb`（默认 20 GB）时拒绝加载。
- **内存超限**：
  - 加载模型前会检查每个 rank 的权重大小，超过 `--mem-budget-gb`（默认为 GPU 推荐工作集的 60%）时拒绝加载。
  - 同时会给 MLX 设置内存和 wired 上限。
  - mlx-lm 默认会把 wired 上限设到接近全部显存，Worker 会把它压回预算以内。
- **GPU 等待网络导致超时**：流水线中的 send/recv/all_gather 全部放在 CPU stream 上，并在提交依赖它们的 GPU 计算之前先 `mx.eval`（见 `swarm/pipeline.py`）。否则在慢链路上会触发 `kIOGPUCommandBufferCallbackErrorTimeout`，严重时会拖死 WindowServer。
- **不要在同一台 Mac 上跑两进程流水线做性能测试**。

## 实测数据

测试环境：M5 MacBook Pro 32 GB（内存带宽 122 GB/s）+ M4 Mac mini 16 GB（101 GB/s），用雷雳 3 线直连（6.5 Gbit/s，往返约 0.26 ms）。

| 模型 | 单机 M5 | 双机流水线（直连线） | 说明 |
|---|---|---|---|
| Qwen3-8B-4bit | 27.9 tok/s | 25.0 tok/s | 理论值的 93.5%，逐 token 一致；走 Wi-Fi 只有 14.7 tok/s |
| Qwen3-32B-4bit | 6.7 tok/s | 6.08 tok/s | 理论值的 91%，逐 token 一致 |
| Qwen3-30B-A3B-4bit (MoE) | 59.1 tok/s | 51.4 tok/s | 逐 token 一致 |
| Qwen3-30B-A3B-6bit (MoE) | 装不下 | 35–37 tok/s | 23 GiB，按 34 + 14 层切分 |

推理池（两台机器都跑 8B-4bit）：
- **交互请求**：8 个并发请求两台各分到 4 个；杀掉一台的 Worker 后，请求全部由另一台成功返回，约 2 秒后被杀的 Worker 自动恢复。
- **批量任务**：40 个任务用时 16 秒。
- **批量任务故障测试**：
  - 运行中 `kill -9` 一台节点并重启 Coordinator：40/40 成功，无重复执行。
  - 节点持有 4 个租约时被 `kill -9`：约 30 秒后这 4 个任务被收回并在另一台完成。

## 开发

```bash
uv run pytest -q
```

### 路线图

- [x] P0：基线测试和流水线可行性验证
- [x] P1：OpenAI 兼容推理池（路由、会话亲和、故障切换）
- [x] P1：批量任务队列（租约、fencing、重试、去重）
- [ ] P1：mDNS 自动发现、launchd 自启动、`swarm top`、更多故障注入
- [ ] P2：把流水线并行接入服务端
- [ ] P3：投机解码、连续批处理

## 安全

当前版本没有鉴权，Worker 和 Coordinator 都监听 `0.0.0.0`，只适合在可信的局域网或直连线上使用。

## 许可证

Apache License 2.0，见 [LICENSE](LICENSE)。
