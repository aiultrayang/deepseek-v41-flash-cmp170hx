# DeepSeek-V4.1-Flash on 4× CMP 170HX — 消费级矿卡跑 671B 混合推理

> 在 4 张 **CMP 170HX**（矿卡，64GB HBM2e，无 P2P、无视频输出）上部署
> **DeepSeek-V4.1-Flash**（FP8 dense + MXFP4 专家 + Engram 表，总权重约 510GB），
> 通过 **CPU/GPU 混合推理**实现 512K 上下文在线服务。

## 硬件

| 部件 | 配置 | 备注 |
|---|---|---|
| GPU | 4× Intel CMP 170HX (SM80 架构，64GB HBM2e) | 矿卡，**无 NVLink/P2P**，单卡约 300W |
| CPU/内存 | AMD 平台，251GB DRAM | CPU 专家计算吃多核 |
| 存储 | /mnt/ssd 2TB NVMe（权重主存）+ /mnt/optane 343GB Intel P5800X 傲腾（Engram 表） | 傲腾的随机读延迟能力用在刀刃上 |

## 核心思路：为什么能跑 510GB 的模型

4×64GB = 256GB 显存，装不下 510GB 权重，靠三层分工：

1. **GPU 层**：FP8 权重的 dense 部分 + 注意力 + 路由器 + shared expert 常驻 HBM
2. **CPU 层**：后 20 层（MoE 层）的 384 个 routed experts（MXFP4/FP4 量化）常驻 DRAM，
   用 CPU 多核做专家 GEMM（`cpu_mxfp4` 内核），GPU 只算路由
3. **Engram 层**：两组 Embedding-Lookup 表（`engram_num_embeddings ≈ 3.84 亿×2`，
   head_dim=256，共 94.42GB）放**傲腾 SSD**，O_DIRECT + Linux AIO 直读，
   不占 page cache、不进 DRAM 常驻

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
  （矿卡无 P2P，TP 代价太高，纯 PP 最优）
- **关键环境变量**：
  ```bash
  NCCL_P2P_DISABLE=1       # 矿卡无 P2P，禁掉后走共享内存/PCIe
  NCCL_ALGO=Ring
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
| `cpu_threads` | `[24,24]` | CPU 专家计算线程分配（两组 NUMA） |
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
   用 `ssd` 模式走 O_DIRECT + Linux AIO（非 GDS），94.42GB 常驻傲腾。
   host workspace 只需要 ~18.2 MiB，GPU workspace 12.4 MiB
2. **Engram 表放傲腾**：`/mnt/optane/model-00047/00048.safetensors` ——
   这两个 94GB 的文件实际是 Engram 表而非"权重"，O_DIRECT 顺序读对傲腾
   的 4K 随机读能力利用得很好
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
| 本方案 | 4×CMP 170HX (256GB) | 510GB（FP8+FP4+Engram） | CPU/GPU 混推 + Engram SSD 分层 |
| 纯 GPU 卸载 | 4×CMP 170HX | 全装显存 | 装不下 510GB，只能跑更小量化 |
| 纯 CPU | 无 GPU | 510GB | 吞吐太低 |

核心洞察：**CMP 170HX 的 64GB HBM2e + 高带宽但无 P2P 的拓扑，
天然适合"PP 流水线 + 专家卸载"而非 TP** ——矿卡的 HBM 容量优势被放大，
P2P 缺陷被 PP 架构规避。

## 待完善

- [ ] 完整的 `start.sh`（当前是生产内脚本，含环境校验/warmup/MPS 清理）
- [ ] `libdsv41_ik` / `libdsv41_cuda` 编译文档
- [ ] 长上下文（128K+）压测数据
- [ ] 傲腾 vs 普通 NVMe 的 Engram 延迟 A/B 对比数据

## 致谢

- DeepSeek 开源 [DeepSeek-V4.1-Flash](https://modelscope.cn/models/deepseek-ai/DeepSeek-V4.1-Flash)
- `buliaoyin/vllm-backport` 的 CPU/GPU 混合推理分支
