# P0 执行手册：基线与可行性验证

目标（见 `doc2-技术评估与开发方案.md` §6 P0）：用实测数据确认路线，产出《基准报告 v1》。

## 0. 每台 Mac 的环境准备

```bash
# 1) 安装 uv（已装可跳过）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2) 拉取本项目到同一路径（多机 mlx.launch 要求各机路径一致，推荐 ~/mlx-swarm）
#    然后安装锁定版本的依赖（mlx 0.32.1 / mlx-lm 0.31.3）
cd ~/mlx-swarm && uv sync
# 国内镜像可选：UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple uv sync

# 3) 多机运行需要：主控机能免密 SSH 到所有节点
ssh-copy-id <user>@<node>
```

**模型准备**：流水线加载只支持 MLX 格式（含 `model.safetensors.index.json`）。用 mlx-community 仓库，或用 `mlx_lm.convert` 转换。
各节点使用同一份模型路径（本地目录或 HF repo 名称）。

国内网络下 Hugging Face（含 hf-mirror）大文件实测只有 10–30 KB/s，基本不可用。改从 ModelScope 的 mlx-community 镜像下载到 `models/`（实测约 4–10 MB/s）：

```bash
mkdir -p models/Qwen3-8B-4bit && cd models/Qwen3-8B-4bit
for f in config.json model.safetensors.index.json tokenizer.json tokenizer_config.json \
         special_tokens_map.json added_tokens.json vocab.json merges.txt model.safetensors; do
  curl -fL -C - -o "$f" "https://modelscope.cn/models/mlx-community/Qwen3-8B-4bit/resolve/master/$f"
done
```

**防死机（本机已死机两次）**：
- `llm_bench.py` 会在加载前检查：每个 rank 的权重不能超过 `--mem-budget-gb`（默认为 GPU 推荐工作集的 60%，M5 32GB 约 15 GB；同一台机器上的多个 rank 平分预算）；磁盘剩余空间不能低于 `--min-free-disk-gb`（默认 20 GB，用于给 swap 留余量）。任何一项不满足就直接拒绝运行，并给 MLX 设置内存硬上限。
- 不要在同一台 Mac 上做两进程流水线的性能测试。两个进程共用一块 GPU，send/recv 时会互相阻塞等待，这是怀疑的死机原因之一。它只用来验证代码路径，0.6B 的正确性已经验证通过。
- 本机磁盘只剩约 39 GB。32B 和 30B-A3B（各 17–18 GB）需要先腾出空间，或者放到其他机器上跑。

## 1. 节点探测（每台都跑）

```bash
uv run python bench/probe.py
```

输出 `results/probe-<host>.json`：芯片、GPU 核数、内存、GPU wired 上限、实测内存带宽和 fp16 TFLOPS、RDMA 状态、雷雳速率、网卡和 IP。

## 2. 网络微基准（每种链路跑一次）

```bash
# 节点 B
uv run python bench/net_bench.py serve
# 节点 A（label 写链路类型：wifi / 1gbe / 2.5gbe / 10gbe / tb4 / tb5）
uv run python bench/net_bench.py client <B 的 IP> --label tb4
```

TB4 直连：用雷雳线连接两台 Mac，在「系统设置 › 网络 › Thunderbolt 网桥」中自动获得 169.254.x.x 地址，或者手动设置同网段的静态 IP。

关注：16 KiB ping-pong 的 p50/p99（解码每跳开销），以及吞吐 Gbit/s（预填充开销）。

## 3. 单机基线（每台、每个模型）

```bash
uv run python bench/llm_bench.py --model mlx-community/Qwen3-8B-4bit --save-ref --label M5-32G
```

- `--save-ref` 写入 greedy 参考输出 `results/ref-<model>.json`，供流水线校验正确性。
- 需要把参考文件拷到流水线 rank 0 所在机器的 `results/` 下。

建议的模型集（按内存选）：

| 用途 | 模型 | 约占内存 |
|---|---|---|
| 链路打通/正确性 | Qwen3-0.6B（bf16）| 1.2 GB |
| dense 基线 | mlx-community/Qwen3-8B-4bit | ~4.5 GB |
| dense 中型 | mlx-community/Qwen3-32B-4bit | ~18 GB |
| MoE 基线 | mlx-community/Qwen3-30B-A3B-4bit | ~17 GB |
| 单机装不下（流水线目标） | 依节点总内存选择，如 70B 4bit / 100B+ MoE 3–4bit | 40 GB+ |

## 4. 流水线（PP）

### 4.1 单机两进程（只验证代码路径，不代表性能）

```bash
uv run mlx.launch --backend ring -n 2 bench/llm_bench.py --model <模型> --label local2
```

### 4.2 多机

`hosts.json`（ring 后端，列表顺序就是流水线顺序：第 0 个持有最前面的层）：

```json
[
  {"ssh": "m5-a.local", "ips": ["192.168.1.10"]},
  {"ssh": "m4-b.local", "ips": ["192.168.1.11"]}
]
```

```bash
uv run mlx.launch --backend ring --hostfile hosts.json \
  bench/llm_bench.py --model <模型> --split w:20,12 --label 2node-1gbe
```

实际可用的写法（M5 + M4 mini，Wi-Fi）：hostfile 见 `hosts-wifi.json`。

- 主控机本身写成 `"ssh": "127.0.0.1"`，launcher 会在本地直接启动，不需要 SSH 到自己。
- 多机模式会直接执行命令。脚本没有可执行权限，所以要显式指定解释器：

```bash
uv run mlx.launch --backend ring --hostfile hosts-wifi.json -- \
  $PWD/.venv/bin/python bench/llm_bench.py --model models/Qwen3-8B-4bit --split w:15,7 --label 2node-wifi
```

**GPU 超时与死机（已修复，见 `swarm/pipeline.py`）**：

- 现象：流水线里的 send/recv/all_gather，以及紧接其后的切片操作，如果放在 GPU stream 上，GPU 命令缓冲区就会等网络。这包括跨 stream 的 event 等待。
- 后果：链路慢的时候（例如 Wi-Fi 传 32 MB 激活值），会触发 `kIOGPUCommandBufferCallbackErrorTimeout`，严重时可能拖死 WindowServer，导致整机卡死。
- 修法：通信全部放在 `mx.cpu` stream 上，并且在提交任何依赖通信结果的 GPU 工作之前，先显式 `mx.eval`。
- 代价：预填充阶段失去了一部分异步重叠。

- `--split`：`even`（平均）、`10,18`（每个 rank 的层数）、`w:20,12`（按权重比例，建议填各节点可用显存 GB）。
- 也可以用 `mlx.distributed_config` 自动生成雷雳环的 hostfile。
- 输出中的 `correctness.identical` 必须为 `true`。

## 5. 对照：exo

在相同节点、相同模型上安装并运行 exo，记录单流和并发 tok/s，作为对照基线。

## 6. Go/No-Go 判定

| 指标 | 通过标准 |
|---|---|
| 正确性 | 流水线 greedy 输出与单机逐 token 一致 |
| 单流解码 | 2 机 PP ≥ 理论值的 70%：理论值 = 1 / Σ(各段权重字节 / 各节点实测带宽) |
| 网络开销 | 有线链路下，PP 解码 tok/s ≥ 同模型单机（若能装下）的 90% |
| 装得下 | 跑通至少一个单机装不下的模型 |

## 7. 结果汇总

所有 JSON 都在 `results/`。最后运行汇总脚本（P0 后半段补充），生成《基准报告 v1》。
