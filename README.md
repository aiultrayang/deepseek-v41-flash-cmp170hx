# DeepSeek-V4.1-Flash on 4× CMP 170HX — 消费级矿卡跑 552B MoE

> 官方口径：**552B 主干参数 + 196B Engram 条件记忆**；每 token 激活参数 **prefill 8B / decode 16B**。
> **CED**（Causal Encoder-Decoder）架构：40 层 = 20 层 causal encoder + 20 层 decoder；
> **CSA2** 稀疏注意力 + **FP4 主 KV 缓存**（E2M1，每 16 通道一个 E4M3 scale）→ 全局 KV 仅 **890 字节/token**（≈V4-Flash 的 1/4）；
> **SWA Bounded Replay** 使持久 KV 占用 ≈ V4-Flash 的 1/8；预训练 45T tokens，原生 1M 上下文。
> 磁盘权重 ~510GB：FP8 dense + FP4 专家（量化打包后张量元素数 < 理论参数量）。

> 在 4 张 **CMP 170HX**（矿卡，64GB HBM2e，单卡 250W，无视频输出）上部署
> **DeepSeek-V4.1-Flash**（FP8 dense + MXFP4 专家 + Engram 表，总权重约 510GB），
> 通过 **CPU/GPU 混合推理**实现 512K 上下文在线服务。
>
> **另有三卡 PP3 方案**（完全复刻 buliaoyin/vllm-backport 官方 README，
> 留出 1 张卡做测试）：见 [docs/pp3-3card-plan.md](docs/pp3-3card-plan.md)。

## 硬件

| 部件 | 配置 | 备注 |
|---|---|---|
| GPU | 4× NVIDIA CMP 170HX（GA100，SM80，64GB HBM2e，合计 256GB 显存） | 矿卡：**无 NVLink，PCIe P2P 可用**（GPU1↔GPU2 同 PHB，其余跨 root complex/NODE），单卡功耗上限 **250W**（最大可调 300W，实测负载仅 40-72W） |
| CPU/内存 | AMD EPYC 7K62 48 核 / 96 线程，单 NUMA，251GB DRAM | 20 个 CPU 专家层的 routed experts 常驻内存，CPU 多核算专家 |
| 存储 | 2TB NVMe（/mnt/ssd，权重分片）+ 343GB Optane SSD（/mnt/optane，Engram 表 ~190GB） | **显存 + 内存 + SSD 三级分层**；Engram 走 Optane O_DIRECT 直读 |

## 核心思路：为什么能跑 510GB 的模型

权重总量约 510GB（552B 主干 + 196B Engram），显存 256GB 装不下，按 **显存 + 内存 + SSD** 三级分工。
能这样分层的前提是官方 CED 架构把每 token 激活压到 **8B（prefill）/16B（decode）**——
每步只用一小撮参数，冷参数按需取，延迟被吞吐摊薄：

| 层级 | 容量 | 常驻内容 |
|---|---|---|
| **显存**（4×64GB HBM2e） | 256GB | FP8 dense 层、注意力、路由器、shared expert + KV cache（~160GB 权重，其余给 KV/工作区） |
| **内存**（251GB DRAM） | ~140GB | 20 个 CPU 专家层 × 384 routed experts（FP4/MXFP4），CPU 多核 GEMM（`cpu_mxfp4` 内核），GPU 只算路由 |
| **SSD**（NVMe + Optane） | ~190GB | **Engram 条件记忆**（官方 196B 参数，按 token 稀疏访问的 lookup 表：`engram_num_embeddings ≈ 3.84 亿×2`，head_dim=256，每元素 ~1B fp8+scale），**傲腾 O_DIRECT + Linux AIO 按需直读**，不占 page cache |

> Engram 表就是权重分片 `model-00047/00048`（软链到 /mnt/optane，各 94.42GiB，
> 合计 ~190GB）。全部 48 个分片（含软链）合计 **475.3 GiB（≈510 GB）**，
> 这是"256GB 显存 + 251GB 内存能跑 510GB 模型"的关键一环。

推理流水线按层切分：`pipeline_layers=[6,6,6,22]`（PP4）。
全部 40 层都是 MoE 层（权重索引实测每层 384 个 routed experts），但
CPU 专家路径只开在层 20-39（`cpu_moe.py` 默认 `start_layer=20`），
且引擎强制要求层 19-39 全部落在最后一个 rank——所以最后 22 层
（层 19 + CPU 专家层 20-39）集中到 PP3，把 CPU 专家计算集中在单 rank：

```
PP0 (GPU0) ── layers 0-5    (GPU 全量层：dense expert GEMM 在卡上)
PP1 (GPU1) ── layers 6-11   (同上)
PP2 (GPU2) ── layers 12-17  (同上)
PP3 (GPU3) ── layers 18-39  (层 19 + CPU 专家层 20-39)
                                └─ 20× CPU MoE 层 routed experts → DRAM (CPU GEMM)
                                └─ Engram tables → Optane NVMe (O_DIRECT)
```

> 这个分层是引擎的硬约束，不是调优选择：`hybrid.py` 校验
> `sum(layers[:-1]) <= 19`，前三个 rank 合计最多 20 层。想均衡负载只能减少 PP 级数
> 或改架构，不是改参数能解决的。

## 软件栈

- **引擎**：`vLLM`（源码编译，`buliaoyin/vllm-backport` 仓库
  `codex/deepseek-v41-cpu-hybrid` 分支），版本 `v0.0.0.dev0`（V1 引擎架构）
- **并行**：`--pipeline-parallel-size 4 --tensor-parallel-size 1`
  （矿卡间 PCIe 拓扑参差：GPU1↔2 走同一 PHB 桥，其余对跨 root complex，
  TP 需要 all-reduce 频繁同步，PP 只在层边界传激活——纯 PP 最优）
- **关键环境变量**（生产实际值）：
  ```bash
  NCCL_P2P_LEVEL=SYS       # P2P 可用（can_device_access_peer=True）；
                           # PCIe 3.0 x16（5GT/s，实测链路即 5GT/s），
                           # 拷贝带宽 ~3.7GB/s；GPU1↔GPU2 走 PHB 直连，
                           # 跨 root complex 对走 SYS 级传输
  NCCL_PROTO=Simple
  VLLM_WORKER_MULTIPROC_METHOD=spawn
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=128   # 稀疏 indexer logits 临时张量上限
  ```
  > 早期尝试过 `NCCL_P2P_DISABLE=1` 强制走共享内存，实测不如放开 P2P。
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
  --speculative-config '{"method":"dspark","num_speculative_tokens":3,"enable_adaptive_verification":true}' \
  --additional-config '{"deepseek_v41_hybrid":{
      "pipeline_layers":[6,6,6,22],
      "cpu_threads":[48,24],
      "engram_cache_gib":8,
      "host_cache_gib":12,
      "engram_storage":"ssd"}}'
```

### additional-config 字段说明

| 字段 | 值 | 含义 |
|---|---|---|
| `pipeline_layers` | `[6,6,6,22]` | 4 个 PP rank 各承载的层数（约束：CPU 专家层 19-39 必须全部落在最后一个 rank，所以前三个 rank 合计最多 20 层） |
| `cpu_threads` | `[48,24]` | **[prefill 线程数, decode 线程数]**——按推理相位运行时热切换（`model_state._set_cpu_phase_threads`：batch 含 prefill 用 48 线程吃满 48 物理核做大矩阵 GEMM，纯 decode 回落 24 线程）；与 NUMA 无关（本机单 NUMA），物理核按 socket/core 拓扑自动探测 |
| `engram_cache_gib` | 8 | Engram 行缓存（GPU 侧 bounded row cache） |
| `host_cache_gib` | 12 | 主存侧 Engram/host cache |
| `engram_storage` | `ssd` | Engram 表走 SSD O_DIRECT（RAM 模式需 ~344GB 主存，本机不够） |

内存水位：SSD 模式启动时实测需要 **~154GB 可用主存**；引擎内置
`check_host_headroom` 会在启动前做主存余量校验（本机门槛 ~236GB MemAvailable，
需要时 `drop_caches` 放行）。

### DSpark 投机解码

官方组件（semi-autoregressive draft generation with confidence-scheduled verification），
`--speculative-config '{"method":"dspark","num_speculative_tokens":3}'`：
vocabulary-sharded Markov head + local argmax reduction，
draft 通信量 O(2×tp_size) 而非 O(vocab_size)，适合 PP 拓扑。
fused Markov step 直接在 GPU 上做（shard width 129280，rank 256，BLOCK_N=256），
不落 `[B, vocab]` 的 logits 大张量。

注意：PP4 + DSpark 组合下，**启动后首轮有约 1/3 概率输出乱码 token**，
需要预热若干个请求（warmup）再对外服务——这是实测出来的行为。

`enable_adaptive_verification:true`（2026-10-17 A/B 实测有效）：
scheduler 侧每步在 K=1/2/3 间自动选档（按期望接受长度/步成本打分，
需超当前档 3% 才切换），confidence head 权重本模型已带。
单流 decode 自动在 K=2/3 间摆动，4 并发时自动降到 K=1（并发步成本
高于单流），等效于给并发场景免去了 2/3 的 draft 验证开销。

DSpark 的收益高度依赖草稿接受率。自然语言文本上接受率远高于随机 token，
两者 decode 速度差近一倍（见下方性能实测）。`num_speculative_tokens`
调大到 7 的参考配置在本机未实测；逐位置接受率 0.65/0.38/0.24 表明
第 4 个之后收益很低，不建议盲调。

## 性能实测（2026-10-17，V4.1-Flash-1017 权重）

### 并发扫描（random 数据集）

`vllm bench serve --dataset-name random`，2048 输入 / 128 输出，`--ignore-eos`，
每档并发数为并发×4 个请求、2 个预热：

| 并发 | 输出吞吐 (tok/s) | Mean TTFT (ms) | Mean TPOT (ms) | DSpark 接受率 |
|---|---|---|---|---|
| 1 | 16.7 | 2,118 | 43.7 | 5.3% |
| 2 | 27.0 | 1,520 | 61.8 | 10.1% |
| 4 | 34.9 | 1,895 | 95.9 | 7.0% |
| 8 | 43.1 | 2,101 | 161.0 | 8.9% |
| 16 | 43.3 | 22,425 | 160.8 | 9.2% |

> **这张表不能当作 decode 速度用。** random 数据集是随机 token 序列，没有语言
> 结构，DSpark 草稿几乎必然被拒（接受率仅 5-10%，接受长度 1.2-1.3），投机解码
> 退化成每步 1 token + 3 个 draft 的验证开销。它只适合看并发扩展性。

### 真实文本 decode（生产口径）

自然语言 prompt 流式统计、稳态窗口去掉前 15%。**2026-10-17 晚优化后**
（`cpu_threads=[48,24]` + `--max-num-batched-tokens 4096` +
`enable_adaptive_verification:true`，DSpark 3 草案）：

| 场景 | 全程 decode | 稳态 decode | TTFT |
|---|---|---|---|
| 单流技术长文（3000 tok） | 50.3-54.6 tok/s | 50.1-54.4 tok/s | — |
| 4 并发（每流 mean） | 24.7-26.4 tok/s | — | 0.6-4.5 s |
| 4 并发（聚合吞吐） | 98.7-105.4 tok/s | — | — |

对比优化前基线（`cpu_threads=[24,24]` + batched 2048 + 固定 K=3）：
单流 50.1-53.4 tok/s（基本持平，单流本就接近 CPU 内存带宽上限），
**4 并发每流 22.9 → 24.7-26.4 tok/s（+8~15%），聚合 91.4 → 98.7-105.4 tok/s**。

> 单流速度瓶颈在 DDR4 内存带宽（~99 GB/s 饱和）与 `gomp_barrier` 自旋，
> 参数层能榨的收益有限；4 并发收益主要来自 adaptive verification 在
> 高并发下自动降到 K=1，减少无效 draft 验证。

### 资源水位

- KV cache 预算 `524,288 tokens`（8 并发上限）；优化后各 PP rank
  `Hybrid cache budget`：KV `[0.295, 0.535, 0.535, 0.852]` GiB、
  expert budget `[10.6, 12.8, 12.6, 24.1]` GiB
- GPU 专家缓存覆盖 43.33%（3328/7680 routed experts，58.3 GiB），
  运行时命中率 78-87%（负载相关），host LRU 11.2 GiB
- 混合 workspace 峰值 ~1.3-1.6 GiB/卡
- 主存峰值 ~165 GB（systemd cgroup 统计）；启动前需 MemAvailable ≥ ~236GB

### decode 瓶颈在哪

单流 decode 时 PP3（承载最后 22 层 + 全部 20 个 MoE 层）GPU 利用率只有 ~65%，
另外三张卡 99-100%；CPU 侧 24 个线程跑满但整机利用率不到 30%（96 线程中的 24 个）。
`perf record` 采样 PP3 进程 8 秒：

```
68.6%  libgomp     gomp_barrier_wait_end        ← 线程同步自旋
 9.6%  libdsv41_ik  dsv41_avx2::project<2>      ← 真正的 MXFP4 GEMM
 5.3%  libdsv41_ik  mul_mat_mxfp4_r8_q8_2_avx2
 2.2%  libdsv41_ik  mul_mat_mxfp4_r8_q8_2_avx2<2>
```

**CPU 线程大部分时间在等同步屏障，不是在算。** 实测本机 8 通道 DDR4-2133
内存带宽多线程 triad 在 8 线程就饱和到 ~99 GB/s，24/48 线程无额外增益——
所以 decode 相位单纯调大 `cpu_threads` 没有意义（A/B 实测 [24,32] 与 [24,8]
均差于 [24,24]，[48,24] 靠 prefill 提速带动 TTFT）。真正的优化方向是
减少 CPU 路由次数（提高 GPU 专家缓存命中）、减少 batch 内的同步次数
和 adaptive verification 降低验证开销，不是加线程。

## 踩坑记录（对复现者最重要）

1. **Engram 存储**：`engram_storage="ram"` 需 ~344GB 主存（251GB 机器不行），
   用 `ssd` 模式走 O_DIRECT + Linux AIO（非 GDS），~190GB 表按需直读傲腾
   （host workspace 仅 ~32.5 MiB，GPU workspace 24.8 MiB，另有 4GiB bounded row cache）
2. **Engram 表放傲腾**：`/mnt/optane/model-00047/00048-of-00048.safetensors`
   （模型目录内以软链形式挂入）——这两个 94.42GiB 的文件（合计 ~190GB）
   实际是 Engram 表而非常规权重，O_DIRECT 按需读，傲腾的低随机读延迟正好吃满
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
8. **别用 random 数据集测 decode 速度**：随机 token 让 DSpark 接受率掉到 5-10%，
   测出来的 16.7 tok/s 不是真实速度，真实文本约 54 tok/s。基准要点见性能实测章节
9. **`cpu_threads` 不是越大越好**：本机内存带宽 8 线程即饱和（~99 GB/s），
   decode 时 CPU 线程 68% 时间在 `gomp_barrier_wait_end` 自旋

## 模型与量化格式

- 权重格式：FP8（ue8m0 scale，block size [32,32]）dense +
  **FP4/MXFP4 专家**（`expert_dtype: fp4`），由 CPU 内核解包
- 架构（官方口径 + 权重索引实测）：**CED** 20 encoder + 20 decoder 层，
  **40 层全部是 MoE**（每层权重里都有 384 routed experts）；
  CPU 混合推理只把层 20-39 的 routed experts 放 DRAM，层 0-19 的
  专家权重仍按 FP4 打包在 GPU 上由 dense 内核计算；
  每 MoE 层 1 shared + 384 routed experts，激活 top-6；
  注意力 **CSA2**（Full/Reindex/Reuse 三静态模式 + Hierarchical Sparse Indexer，
  compressor/indexer 仅在层 2/8/14/20 与 indexer 源层 24/28/32/36 出现），
  **FP4 主 KV 缓存**（E2M1，每 16 通道一个 E4M3 scale）→ 全局 KV 890 B/token；
  另有 Single-Pass mHC（Mega-mHC 内核）；Engram 挂在层 1/14
- 原生上下文 1M（预训练 64K 稀疏注意力，34T tokens 处扩到 1M），
  YaRN 16×（original 65536），本部署开放 512K
- Vision：DeepSeek-ViT（2D-RoPE + 3×3 pixel-unshuffle 下采样）+ 两层 MLP projector，
  从预训练起即图文联合训练；支持图片理解（每请求 1 张）

## 与其他矿卡方案对比

| 方案 | 卡 | 权重 | 特点 |
|---|---|---|---|
| 本方案 | 4×CMP 170HX (256GB) | ~510GB（FP8+FP4+Engram） | 显存+内存+SSD 三级分层，PP4 |
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
- [x] DSpark `enable_adaptive_verification` A/B（2026-10-17 完成，4 并发 +8~15%）
- [x] 三卡 PP3 复刻方案与资源/速度总结（2026-10-18，见 [docs/pp3-3card-plan.md](docs/pp3-3card-plan.md)）
- [ ] DSpark `num_speculative_tokens` 3/5/7 扫描（接受率 0.65/0.38/0.24，预期收益低）
- [ ] `route_tail_skip=4/5`（有损加速，参考配置宣称 decode +15~24%）的实测

## 致谢

- DeepSeek 开源 [DeepSeek-V4.1-Flash](https://modelscope.cn/models/deepseek-ai/DeepSeek-V4.1-Flash)
- `buliaoyin/vllm-backport` 的 CPU/GPU 混合推理分支
