# 三卡 PP3 方案（完全复刻 buliaoyin/vllm-backport 官方 README）

> 2026-10-18 在同一台 4×CMP 170HX 机器上，将 PP4 四卡方案切换为
> [buliaoyin/vllm-backport](https://github.com/buliaoyin/vllm-backport)
> README 的三卡官方方案（PP3 `[7,8,25]`），留出 GPU3 做单卡实验。
> 本文记录复刻要点、资源分配与实测速度，与[四卡方案](README.md)对照。

## 配置对照：PP3 vs PP4

| 项 | PP4（四卡，见主 README） | PP3（三卡，本文） |
|---|---|---|
| GPU | 4 张（GPU0-3） | **3 张（GPU0-2），GPU3 留空** |
| `--pipeline-parallel-size` | 4 | **3** |
| `pipeline_layers` | `[6,6,6,22]` | **`[7,8,25]`** |
| `cpu_threads` | `[48,24]` | **`[24,24]`**（官方 README 原值） |
| `NCCL_P2P_DISABLE` | 未设（开 P2P，实测更优） | **=1**（官方 README 原值） |
| `--kv-cache-tokens` | 524288 | 524288（官方为 1048576，对应 1M 上下文场景；本机 512K maxlen 维持等效） |
| DSpark | 3 + adaptive_verification | 同（adaptive 为本机增量，官方示例未开） |
| Engram | ssd + `engram_cache_gib=8` + `host_cache_gib=12` | 同（官方 SSD 示例默认行缓存仅 1 GiB，本机 8 GiB 为增量调优） |
| `--max-num-batched-tokens` | 4096 | 同 |
| maxlen / util / seqs | 524288 / 0.97 / 8 | 同 |

启动命令与主 README 相同，仅替换上述参数。GPU3 完全空闲（52 MiB），可用于
单卡小模型实验或故障卡轮换。

## 20 层 CPU 专家的分布：PP3 与 PP4 的本质差异

DeepSeek-V4.1-Flash 的 CPU 混合路径只开在层 20-39（`cpu_moe.py`
默认 `start_layer=20`），引擎强制这些层全部落在最后一个 rank：

```
PP3 [7,8,25]:
  PP0 (GPU0) ── layers 0-6    纯 GPU dense 段
  PP1 (GPU1) ── layers 7-14   纯 GPU dense 段
  PP2 (GPU2) ── layers 15-39  含全部 20 个 CPU MoE 层（日志实测
                                20 条 "CPU routed experts" 初始化全在 PP2）
                                └─ 20×384 routed experts → DRAM (CPU GEMM, 134.5 GiB)
                                └─ Engram tables → Optane (O_DIRECT)

PP4 [6,6,6,22]:
  PP3 (GPU3) ── layers 18-39  同样独扛全部 20 个 CPU MoE 层
```

两方案的 CPU 专家总量相同（20 层 × 384 专家 = 134.5 GiB 常驻 DRAM），
都压在最后一个 rank。差别在前段：PP3 把 40 层摊给 2 个纯 GPU rank，
PP4 摊给 3 个——**GPU 段数量不同，但 CPU GEMM 关键路径完全一样**。

## 资源分配（实测，2026-10-18）

### GPU 显存（每卡 64 GiB，util 0.97）

| 卡 | 分段 | 实测占用 | 内部构成 |
|---|---|---|---|
| GPU0 | PP0 layers 0-6 + Engram 表1 | 60.5 GiB | 分段权重 + KV 7.31 GiB + Engram 行缓存 4 GiB + workspace |
| GPU1 | PP1 layers 7-14 + Engram 表2 | 61.6 GiB | 同上 |
| GPU2 | PP2 layers 15-39 | 60.8 GiB | 分段权重 + KV 7.31 GiB + 混合 workspace |
| GPU3 | 空闲 | 52 MiB | 留作测试 |

- KV cache：fp8_ds_mla 格式，block=128，预算 **524,288 tokens**（8 请求共享），
  每 rank 7.31 GiB
- Engram GPU workspace 24.8 MiB；bounded row cache 每表 4 GiB
  （`engram_cache_gib=8` 平分给 PP0/PP1 两张表）

### 主存（251 GB，运行中 MemAvailable ~73 GiB）

| 进程 | PSS | 内容 |
|---|---|---|
| Worker_PP2 | **150.3 GiB** | 20 层 CPU 专家 134.5 GiB（每层 384×17.9 MiB）+ `host_cache_gib` 12 GiB + 其余 |
| Worker_PP0/PP1 | 各 6.5 GiB | dense 段权重 + 各 4 GiB Engram 行缓存 |
| EngineCore + APIServer | ~1 GiB | 调度 |

启动门槛与 PP4 相同：`check_host_headroom` 要求
**MemAvailable ≥ 234.47 GiB**（64 保守余量 + 134.5 CPU 专家 + 12 host 缓存
+ 16 DSpark/adaptive 抬升）。20 层 CPU 专家全部在 PP2，与卡数无关，
所以 page cache 高时仍需 `drop_caches`（本机实测 236→240 GiB 后一次通过）。

### SSD/NVMe

| 盘 | 用途 |
|---|---|
| /mnt/ssd（2TB NVMe） | 模型 476 GiB（48 shards）主副本 |
| /mnt/optane（傲腾 343GB） | **Engram 表 2×94.42 = 188.8 GiB**，O_DIRECT + Linux AIO 直读（非 GDS），host workspace 仅 32.5 MiB |

三级分层结构与 PP4 完全一致，只是显存从 4 卡均摊变 3 卡加重。

## 实测速度（2026-10-18，同口径）

### 与 PP4 同机对照

| 场景 | PP4（4 卡） | PP3（3 卡） | 比值 |
|---|---|---|---|
| 单流 decode（稳态，引擎口径） | 43-61 tok/s | **27-31 tok/s** | ~55% |
| 4 并发每流 | 24.7-26.4 tok/s | **12.8 tok/s** | ~52% |
| 4 并发聚合 | 98.7-105.4 tok/s | **52-62 tok/s** | ~55% |

（PP3 启动 ~4 分钟就绪；冒烟正常；PP4 数据来自主 README 2026-10-17 实测）

### 与参考项目官方实测对照

参考项目（fork SSD 文档，2026-09-18，3×170HX + EPYC 7532 双路 + 企业 NVMe，
32K 输入/256 输出、DSpark3）：

| 项 | 参考项目 | 本机 | 说明 |
|---|---|---|---|
| 单流 decode（RAM 模式） | 29.5 tok/s | 27-31 tok/s | **持平** |
| 单流 decode（SSD 模式） | 29.1 tok/s | 27-31 tok/s | decode 对存储模式不敏感，两边一致 |
| prefill（SSD+128MiB 行缓存） | 3,207 tok/s | ~1.5-2.9K tok/s | 本机偏低：测试口径 15.4K vs 32K 输入、傲腾 vs 企业 NVMe 随机读 |
| prefill（SSD+1GiB 行缓存） | 5,782 tok/s | — | 行缓存拉满可追平 RAM；本机 8 GiB 行缓存热态应更优 |
| prefill（RAM） | 5,690 tok/s | 不适用 | 本机 251G 主存跑不了 RAM 模式 |

### 为什么 3 卡比 4 卡慢约一半

差距**不在卡数本身**，在 CPU GEMM 的重叠窗口：

- **PP4 `[6,6,6,22]`**：单步 decode 链路为 PP0→PP1→PP2→PP3，两个 CPU 段
  （PP2/PP3 各约 10 层 CPU 专家）可以与 GPU 段**流水重叠**，
  单步时延里 CPU GEMM 有效只算一份。
- **PP3 `[7,8,25]`**：PP2 独扛 20 层 CPU GEMM **全部串行**，
  前面两个 rank 是纯 GPU 段，CPU 计算时其余卡在等——20 层 GEMM 时间
  全额进关键路径，没有任何重叠窗口。

内存墙两边相同（DDR4 ~99 GB/s 饱和，decode 线程 24 两边一致，
PP4 时代实测加线程无收益），所以单段 GEMM 速度一致、总时延差约 2 倍。

**结论**：三卡方案省出 1 张卡的代价是吞吐减半。decode 单流仍可达
27-31 tok/s（追平参考项目官方水平），适合"必须留卡"的场景；
追求吞吐应回 PP4（配置在 `start-dsv41-18050.sh.pp4.bak`）。

## 复现要点

1. 启动脚本（生产版，含预检/日志/PID 管理）：
   `GPUS=0,1,2`、`--pipeline-parallel-size 3`、
   `--additional-config '{"deepseek_v41_hybrid":{"pipeline_layers":[7,8,25],
   "cpu_threads":[24,24],"engram_cache_gib":8,"host_cache_gib":12,
   "engram_storage":"ssd"}}'`
2. 其余参数（KV 预算、DSpark、parsers、template kwargs）与主 README 启动命令一致
3. `NCCL_P2P_DISABLE=1` 按官方 README 保留（PP4 时代本机实测开 P2P 更优，
   三卡方案未复测该对比，先用官方值）
4. 启动前 `MemAvailable` 不足时先 `drop_caches`（门槛 234.47 GiB）
5. 切换前先彻底停旧实例：`pkill -f "vllm.*--port 18050"`，确认 GPU 显存释放
   （`nvidia-smi` 全部 ~52 MiB）再启动，否则 "Address already in use"

## 待完善

- [ ] PP3 下 DSpark adaptive_verification 的接受率与档位行为（未采样）
- [ ] PP3 长上下文（128K+）TTFT 数据
- [ ] `NCCL_P2P_DISABLE=1` vs 开 P2P 在 PP3 下的 A/B
- [ ] GPU3 留卡的单卡实验（EXL3 小模型 / 短上下文 DSV4F）实测
