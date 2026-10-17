# DeepSeek-V4.1-Flash on 4× CMP 170HX — 消费级矿卡跑 552B MoE

> 官方口径：**552B 主干参数 + 196B Engram 条件记忆**；每 token 激活参数 **prefill 8B / decode 16B**（CED 架构，top-6 路由）。
> 磁盘权重 ~510GB：FP8 dense + FP4 专家（量化打包后张量元素数 < 理论参数量）。

> 在 4 张 **CMP 170HX**（矿卡，64GB HBM2e，单卡 250W，无视频输出）上部署
> **DeepSeek-V4.1-Flash**（FP8 dense + MXFP4 专家 + Engram 表，总权重约 510GB），
> 通过 **CPU/GPU 混合推理**实现 512K 上下文在线服务。

## 硬件

| 部件 | 配置 | 备注 |
|---|---|---|
| GPU | 4× Intel CMP 170HX（SM80，64GB HBM2e，合计 256GB 显存） | 矿卡：**无 NVLink，PCIe P2P 可用**（GPU1↔GPU2 同 PHB，其余跨 root complex/NODE），单卡功耗上限 **250W** |
| CPU/内存 | AMD EPYC 7K62 48 核 / 96 线程，单 NUMA，251GB DRAM | 后 20 层 routed experts 常驻内存，CPU 多核算专家 |
| 存储 | 2TB NVMe（/mnt/ssd，权重分片）+ 343GB Intel Optane P5800X（/mnt/optane，Engram 表 ~190GB） | **显存 + 内存 + SSD 三级分层**；Engram 走傲腾 O_DIRECT 直读 |

## 核心思路：为什么能跑 510GB 的模型

权重总量约 510GB，显存 256GB 装不下，按 **显存 + 内存 + SSD** 三级分工：

| 层级 | 容量 | 常驻内容 |
|---|---|---|
| **显存**（4×64GB HBM2e） | 256GB | FP8 dense 层、注意力、路由器、shared expert + KV cache（~160GB 权重，其余给 KV/工作区） |
| **内存**（251GB DRAM） | ~140GB | 后 20 个 MoE 层的 384 个 routed experts（FP4/MXFP4），CPU 多核 GEMM（`cpu_mxfp4` 内核），GPU 只算路由 |
| **SSD**（NVMe + Optane） | ~190GB | 两组 Engram lookup 表（`engram_num_embeddings ≈ 3.84 亿×2`，head_dim=256，每元素 ~1B fp8+scale），**傲腾 O_DIRECT + Linux AIO 按需直读**，不占 page cache |

> Engram 表就是权重分片 `model-00047/00048`（各 94.42GiB，合计 ~190GB），
> 这是"256GB 显存 + 251GB 内存能跑 510GB 模型"的关键一环。

推理流水线按层切分：`pipeline_layers=[6,6,7,21]`（PP4），
其中前三个 rank 覆盖 6+6+7=19 层dense+注意力，最后一个 rank 承载 21 层
（含全部 20 个 CPU-MoE 层）。

```
PP0 (GPU0) ── layers 0-5    (dense/attention)
PP1 (GPU1) ── layers 6-11   (dense/attention)
PP2 (GPU2) ── layers 12-18  (dense/attention)
PP3 (GPU3) ── layers 19-39  (attention + 20× CPU-MoE 层)
                                └─ routed experts → DRAM (CPU GEMM)
                                └─ engram tables → Optane NVMe (O_DIRECT)
```

## 软件栈

- **引擎**：`vLLM`（源码编译，`buliaoyin/vllm-backport` 仓库
  `codex/deepseek-v41-cpu-hybrid` 分支），版本 `v0.0.0.dev0`（V1 引擎架构）
- **并行**：`--pipeline-parallel-size 4 --tensor-parallel-size 1`
  （矿卡间 PCIe 拓扑参差：GPU1↔2 走同一 PHB 桥，其余对跨 root complex，
  TP 需要 all-reduce 频繁同步，PP 只在层边界传激活——纯 PP 最优）
- **关键环境变量**：
  ```bash
  NCCL_P2P_DISABLE=1       # 实测 P2P 拷贝仅 3.7GB/s（PCIe 3.0 x16 单向瓶颈）
  NCCL_ALGO=Ring           # 且跨 root 的 P2P 要绕路，收益不稳定，统一走共享内存稳妥
  NCCL_PROTO=Simple
  VLLM_WORKER_MULTIPROC_METHOD=spawn
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=128   # 稀疏 indexer logits 临时张量上限
  ```
- **CUDA MPS**：引擎自动托管 `/tmp/dsv41-mps-*` 实例，
  `CUDA_DEVICE_MAX_CONNECTIONS=16`
- **原生内核**：需要自编译 `libdsv41_ik.so` + `libdsv41_cuda.so`
  （CMake targets `libdsv41_ik`、`libdsv41_cuda`），upstream 预编译 wheel 不带

## 启动命令（生产参数）

```bash
vllm serve /mnt/ssd/models/DeepSeek-V4.1-Flash \
  --served-model-name DeepSeek-V4.1-Flash \
  --host 127.0.0.1 --port 18050 \
  --pipeline-parallel-size 4 --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.97 \
  --max-num-seqs 8 \
  --max-num-batched-tokens 2048 \
  --max-model-len 524288 \
  --kv-cache-tokens 524288 \
  --reasoning-parser deepseek_v41 \
  --tool-call-parser deepseek_v41 \
  --enable-auto-tool-choice \
  --default-chat-template-kwargs '{"reasoning_effort":50}' \
  --speculative-config '{"method":"dspark","num_speculative_tokens":3}' \
  --additional-config '{"deepseek_v41_hybrid":{
      "pipeline_layers":[6,6,7,21],
      "cpu_threads":[24,24],
      "engram_cache_gib":8,
      "host_cache_gib":12,
      "engram_storage":"ssd"}}'
```

### additional-config 字段说明

| 字段 | 值 | 含义 |
|---|---|---|
| `pipeline_layers` | `[6,6,7,21]` | 4 个 PP rank 各承载的层数 |
| `cpu_threads` | `[24,24]` | **[prefill 线程数, decode 线程数]**——按推理相位运行时热切换（`model_state._set_cpu_phase_threads`：batch 含 prefill 用 24 线程吃满多核对大矩阵 GEMM 的收益，纯 decode 回落 24 线程小 batch 防空转）；与 NUMA 无关（本机单 NUMA），物理核按 socket/core 拓扑自动探测 |
| `engram_cache_gib` | 8 | Engram 行缓存（GPU 侧 bounded row cache） |
| `host_cache_gib` | 12 | 主存侧 Engram/host cache |
| `engram_storage` | `ssd` | Engram 表走 SSD O_DIRECT（RAM 模式需 ~344GB 主存，本机不够） |

内存水位：SSD 模式启动时实测需要 **~154GB 可用主存**；引擎内置
`check_host_headroom` 会在启动前做主存余量校验（本机门槛 ~236GB MemAvailable，
需要时 `drop_caches` 放行）。

### DSpark 投机解码

`--speculative-config '{"method":"dspark","num_speculative_tokens":3}'`：
vocabulary-sharded Markov head + local argmax reduction，
draft 通信量 O(2×tp_size) 而非 O(vocab_size)，适合 PP 拓扑。
fused Markov step 直接在 GPU 上做（shard width 129280，rank 256，BLOCK_N=256），
不落 `[B, vocab]` 的 logits 大张量。

注意：PP4 + DSpark 组合下，**启动后首轮有约 1/3 概率输出乱码 token**，
需要预热若干个请求（warmup）再对外服务——这是实测出来的行为。

## 性能实测（2026-09-30 基准）

固定 16K input / 2K output，`request_rate=inf` 突发模式：

| 并发 | 总输入 tok | 总输出 tok | 输出吞吐 (tok/s) | 总吞吐 (tok/s) |
|---|---|---|---|---|
| 1 | 16,384 | 1,024 | 15.3 | 260.0 |
| 2 | 16,384 | 1,024 | 23.1 | 392.6 |
| 4 | 32,768 | 2,048 | 26.9 | 457.6 |
| 8 | 65,536 | 4,096 | 34.4 | 585.4 |
| 16 | 131,072 | 8,192 | 37.7 | 640.4 |

- **单流 decode**：~15-20 tok/s（PP 流水线深度限制，首 token 延迟受流水线影响）
- **总吞吐随并发稳定增长**，16 并发时接近 640 tok/s
- KV cache：`524,288 tokens` 预算（8 并发上限），实际池等效 543,619 tokens
- GPU 显存水位：0.97 utilization，混合 workspace 峰值 ~0.682 GiB/卡

## 踩坑记录（对复现者最重要）

1. **Engram 存储**：`engram_storage="ram"` 需 ~344GB 主存（251GB 机器不行），
   用 `ssd` 模式走 O_DIRECT + Linux AIO（非 GDS），~190GB 表按需直读傲腾
   （host workspace 仅 ~18.2 MiB，GPU workspace 12.4 MiB，另有 4GiB bounded row cache）。
   host workspace 只需要 ~18.2 MiB，GPU workspace 12.4 MiB
2. **Engram 表放傲腾**：`/mnt/optane/model-00047/00048.safetensors` ——
   这两个 94.42GiB 的文件（合计 ~190GB）实际是 Engram 表而非常规权重，
   O_DIRECT 按需读，傲腾的低随机读延迟正好吃满
3. **Warmup 必开**：PP4+DSpark 首轮请求乱码问题，启动后发 8 个预热请求解决
4. **MPS 残留**：每次 PP4 启动会派生 `nvidia-cuda-mps-server`，
   旧实例不自动退出（每轮吃 1~1.5GB 主存），重启脚本要显式清理
5. **孤儿进程**：vLLM 多进程模型下 kill API server 不会带走
   EngineCore/Worker，需要 `pkill -f "VLLM::EngineCore"` +
   `pkill -f "VLLM::Worker_PP"` 一并清理
6. **启动前主存校验**：`check_host_headroom` 要求 MemAvailable ≥ 门槛
   （本机 ~236GB），page cache 会吃掉余量，必要时 `drop_caches`
7. **API 走 Anthropic 协议也行**：`/v1/messages` 同样可用，
   reasoning 模型 `max_tokens` 建议不低于 512，否则推理段占满、正文为空

## 模型与量化格式

- 权重格式：FP8（ue8m0 scale，block size [32,32]）dense +
  **FP4/MXFP4 专家**（`expert_dtype: fp4`），由 CPU 内核解包
- 架构：40 层，`hidden_size=5120`，384 routed experts + 1 shared expert，
  top-6 routing（`scoring_func=sqrtsoftplus`），MLA 注意力
  （`qk_rope_head_dim=64`，head_dim=512）
- 原生上下文 1M（`max_position_embeddings=1048576`），
  YaRN 16×（original 65536），本部署开放 512K
- Vision：`image_token_id=129264`，支持图片理解（每请求 1 张）

## 与其他矿卡方案对比

| 方案 | 卡 | 权重 | 特点 |
|---|---|---|---|
| 本方案 | 4×CMP 170HX (256GB) | 510GB（FP8+FP4+Engram） | 显存+内存+SSD 三级分层，PP4 |
| 纯 GPU 卸载 | 4×CMP 170HX | 全装显存 | 装不下 510GB，只能跑更小量化 |
| 纯 CPU | 无 GPU | 510GB | 吞吐太低 |

核心洞察：**CMP 170HX 的 64GB HBM2e 大显存 + 250W 低功耗，
天然适合"PP 流水线 + 显存/内存/SSD 三级专家分层"而非 TP** ——
矿卡的容量优势被放大，PCIe 拓扑参差与 P2P 带宽瓶颈被 PP 架构规避。

## 待完善

- [ ] 完整的 `start.sh`（当前是生产内脚本，含环境校验/warmup/MPS 清理）
- [ ] `libdsv41_ik` / `libdsv41_cuda` 编译文档
- [ ] 长上下文（128K+）压测数据
- [ ] 傲腾 vs 普通 NVMe 的 Engram 延迟 A/B 对比数据

## 致谢

- DeepSeek 开源 [DeepSeek-V4.1-Flash](https://modelscope.cn/models/deepseek-ai/DeepSeek-V4.1-Flash)
- `buliaoyin/vllm-backport` 的 CPU/GPU 混合推理分支
