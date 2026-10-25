# PP6 + Engram-SSD 方案（8× CMP 170HX 只用 6 卡）

> 与根目录 README 的 **PP4 + 内存 + 傲腾 SSD** 方案并列的第二套打法。
> 同一模型、同一代矿卡，但分层哲学完全相反：**PP4 把 CPU 内存当第二级存储**（20 层 CPU 专家）, 
> **PP6 把 CPU 完全排除在计算之外**，把放不下的 Engram 表直接从 NVMe 流式喂给 GPU。

| | PP4（根目录 README） | **PP6 + SSD（本文）** |
|---|---|---|
| 卡数 | 4（全部用上） | **6（8 卡机只用 6 张）** |
| 显存 | 256 GB | 384 GB |
| 第二级存储 | DRAM 140 GB（CPU 专家层）+ 傲腾 190 GB | **无内存层**，Engram 189 GiB 走 NVMe |
| 计算分工 | 20 层 routed experts 在 CPU（AVX2 MXFP4） | **全部层在 GPU**，CPU 只做调度 |
| 引擎 | `buliaoyin/vllm-backport` `codex/deepseek-v41-cpu-hybrid` 分支（v0.0.0.dev0） | **vLLM 0.13.1**（`344303947/dsv41-flash-pp5-170hx@d63af5a` 源码 + 补丁） |
| 层切分 | `[6,6,6,22]` | **`7,7,7,7,7,5`** |
| 单流 decode | 50–54 tok/s | **140 tok/s**（热缓存，counting@500） |
| 并发 c8 | 4 并发 98–105 tok/s 聚合 | **754.9 tok/s 聚合（每客户端 92.2）** |
| 适用场景 | 只有 4 张卡、要 512K 长上下文在线服务 | 有 6–8 张卡、要吞吐与并发 |

---

## 一句话概括

**6 张 64 GB 的 CMP 170HX（合计 384 GB 显存）装不下 475 GiB 的 DeepSeek-V4.1-Flash，于是把占 189 GiB 的两层 Engram 表直接从 SSD 用 `O_DIRECT + Linux AIO` 流式喂给 GPU（8 GiB 热点行缓存），其余权重按 PP6（`7,7,7,7,7,5`）切分常驻显存，dspark 投机解码 5 token 提速。**

```
        ┌────────────── NVMe ────────────────────────┐
        │ /srv/dsv41/engram/  189 GiB ← shard 47/48  │
        └───────────────────┬────────────────────────┘
                            │ O_DIRECT + AIO (QD128, 4K 页, 1024 页/批)
                            ▼
  GPU0 ─ GPU1 ─ GPU2 ─ GPU3 ─ GPU4 ─ GPU5      ← PP6 串行流水 (layers 7,7,7,7,7,5)
   │      │      │      │      │      │
   └──── NCCL Ring / Proto=Simple / P2P 关 ────┘   ← PCIe Gen2 ×4，无 NVLink
                            ▲
        ┌───────────────────┴────────────────────────┐
        │ SATA 2T：其余 46 个 shard（≈286 GiB）lazy 读 │
        └────────────────────────────────────────────┘
                            │
                     http://0.0.0.0:8001
```

---

## 1. 硬件与软件栈（实测机器）

| 项目 | 规格 | 对部署的影响 |
|---|---|---|
| GPU | 8× CMP 170HX（`nvidia-smi` 65536 MiB/卡，已解锁），**实际只用 GPU0–5** | 384 GB 显存仍装不下 475 GiB → Engram 必须落盘 |
| 架构 | SM80（GA100 阉割版），**无 NVLink、无 P2P** | 不能 TP8；PP 通信走 PCIe |
| PCIe | **Gen2 ×4**（实测 `link.gen=2, width=4`，约 2 GB/s/卡） | 激活跨卡传输是 PP6 的第二瓶颈 |
| CPU / RAM | 32 线程 / **125 GB** | 放不下 189 GiB Engram，也无法给 PP worker 留带宽 → 走 SSD |
| 系统盘 | 915 G NVMe，已用 92%（Engram 189 G 在这） | ⚠️ 只剩 74 G，需监控 |
| 数据盘 | 2.0 T SATA，已用 76%（完整 checkpoint） | lazy 加载 + 页缓存靠它 |
| OS | Ubuntu 24.04 | — |
| 镜像 | `dsv41-pp6-engram-ssd:port1`，29.7 GB | 基础镜像 `docker.io/lazymio/vllm-backport@sha256:8094fcbab905…` |
| 运行时 | vLLM **0.13.1**（源码 overlay）+ torch **2.13.0+cu130** + CUDA 13.0.3 | 官方 pip 的 vLLM 没有 `deepseek_v4_1` 包，必须自建 |
| 模型 | DeepSeek-V4.1-Flash，48 shard，fp8 block 32×32 dense + expert fp4，40 层，384 routed experts | — |

### 模型体积账（为什么必须 SSD）

| 组成 | 体积 |
|---|---|
| 常规权重（shard 1–46） | ≈ 286 GiB |
| **Engram 表 ×2（shard 47/48，挂在层 1 和 14）** | **≈ 189 GiB** |
| 合计 | ≈ 475 GiB |
| PP6 可用显存（6×64 GB，扣 KV/激活） | ≈ 384 GB |

差值 ~90 GiB 无处存放 → `--engram-config '{"cpu_offload":true,"storage":"ssd","engram_cache_gib":8}'`：
Engram 表既不进显存也不进 RAM，直接从盘读。

> 注意与 PP4 的差别：PP4 的 `engram_storage="ram"` 需要 ~344 GB 主存（251 GB 机器不够，所以也退到 `ssd`）。
> PP6 机器只有 125 GB，**从一开始就只能是 ssd**。

---

## 2. 服务器端目录布局

```
/srv/dsv41/
├── model/DeepSeek-V4.1-Flash/        # 全部符号链接 → 真实 checkpoint
│   ├── config.json → <CKPT_ROOT>/DeepSeek-V4.1-Flash-1017/config.json
│   ├── model-00001…00046-of-00048.safetensors → <CKPT_ROOT>/model/…
│   └── model-00047/00048-of-00048.safetensors → /srv/dsv41/engram/…   ← 关键：Engram 走 NVMe
├── engram/                           # 189 GiB，两个 Engram shard（O_DIRECT 读这里）
├── cache/                            # 挂进容器 /root/.cache（HF/TMPDIR），避免容器层膨胀
└── port/                             # 构建工作区
    ├── src/                          # git 检出 344303947/dsv41-flash-pp5-170hx @ d63af5a
    ├── build-port.sh                 # 一键组装 + 构建（本仓库 build/build-port.sh）
    ├── Containerfile                 # 两阶段：native .so + python overlay
    ├── compose.yml                   # 运行编排（含完整 vllm serve 命令）
    ├── *.patch                       # 两个补丁（本仓库 patches/）
    ├── context/native/engram_ssd.cpp # O_DIRECT + AIO + 有界行缓存（本仓库 native/）
    └── verify_ssd.py                 # 正确性（乱码判据）+ 吞吐验收
```

**为什么 shard 47/48 要单独软链到 NVMe**：这两个 94.42 GiB 的文件其实是 Engram 表而非常规权重，
需要按 token 随机读（O_DIRECT）；其余 46 个 shard 只读一次，留在 SATA 即可，两条 I/O 不抢带宽。

---

## 3. 镜像构建

```bash
# 0. 把本目录 build/ 与 patches/ native/ 拷到 /srv/dsv41/port 下
#    目录结构见 build-port.sh 里的 $R
bash build-port.sh
# 期望末行：BUILD_DONE dsv41-pp6-engram-ssd:port1
```

`build-port.sh` 的五步：① `git fetch --depth 1` 取出 `344303947/...@d63af5a`；
② `git apply --check` + apply 两个补丁，然后 `py_compile` 六个改动文件 + `grep -q _pp_communicators_primed` 守门；
③ 组装构建上下文（只 `cp -a src/vllm`，源码树不进镜像）；④ 两阶段 build；
⑤ 起容器验 `.so` 可 `ctypes.CDLL` 加载并打印版本。

**Containerfile 两阶段要点**

- stage `native`：现场装 g++，自动探测 `cuda_runtime_api.h` / `libcudart.so` 位置（系统 CUDA 与 pip 的
  `nvidia/cu*/include` 两种布局都兼容）；若只有带版本号的 `libcudart.so.13`，则软链兜底
  `-l:libcudart.so -Wl,-rpath`；编译完用 `nm -D` 验导出符号 `dsv41_engram_ssd_create`、`ldd` 验 cudart。
- stage 2：只 `COPY vllm/`（纯 Python overlay）+ 那个 `.so` → 镜像 29.7 GB 主要是基础镜像。
- 打标签 `io.canozziia.upstream` / `io.workbuddy.patch` 便于溯源。

---

## 4. 两个补丁做了什么

### 4.1 `0001-prime-pp-communicators-before-model-load.patch`（PP 通信器预热，3.3 KB）

**问题**：vLLM 的 PP 点对点通信器是**惰性创建**的 —— 首次 `pp_group.send/recv` 才分配 NCCL 通信器
（6 卡 PP 是 5 对 rank × 2 方向，各占显存）。而显存 profiling 发生在模型加载后、推理前，
此时通信器还不存在 → profiling **低估**峰值显存 → 上真负载首次建通信器时 OOM 或剧烈抖动。
症状是「能起来，但一压就炸」，极难归因。

**做法**：在 `GPUModelRunner` 构造之后、模型/KV 分配之前调用 `_prime_pp_communicators()`，
向每对 rank 发/收一个 `int32[1]` primer tensor，把通信器提前建好；
`getattr(self, "_pp_communicators_primed", False)` 保证幂等，`world_size <= 1` 短路。

### 4.2 `engram-ssd-port.patch`（Engram SSD 后端移植，38 KB / 6 文件）

| 文件 | 改动 |
|---|---|
| `csrc/cpu/dsv41/engram_ssd.cpp`（新增） | C++ 后端：`O_DIRECT` 打开 safetensors、Linux `io_setup/io_submit/io_getevents` AIO（队列深 128、4 KB 页、1024 页/批）、有界热点行缓存、CUDA stream 异步拷贝、C ABI `dsv41_engram_ssd_{create,enqueue,stats,clear_cache,destroy,version}` |
| `vllm/config/engram.py` | `EngramConfig.storage` 增加 `"ssd"`；新增 `engram_cache_gib`；**校验 `storage=="ssd"` 必须 `tensor_parallel_size==1`**；校验 load_format 与 lazy 加载 |
| `vllm/models/deepseek_v4_1/common/engram_ssd.py`（新增） | Python 侧：表切分/几何推导、`(layer, row)` → 文件偏移、缓存淘汰前等 CUDA 完成 |
| `vllm/models/deepseek_v4_1/common/engram.py` | 按 storage 分流：`ram` 走 host memory over UVA，`ssd` 走原生后端 |
| `vllm/models/deepseek_v4_1/host_memory.py` | SSD 模式不再分配那 189 GiB 主机内存（否则 125 GB 机器直接 OOM） |
| `vllm/config/vllm.py` | 把 engram 配置接到并行/加载策略校验 |

**核心设计**：SSD 模式**直接读原始 checkpoint 切片，不做 ETL 转格式** —— 省掉一次 189 GiB 的全量转换。

**显存账的真相**：`ParallelEngramEmbedding.__init__` 在 SSD 模式下用 `device="meta"` 占位
（零分配），并把 `weight_loader` 换成 `ssd.load`（只登记「文件哪个偏移、多少行」，不拷贝）。
于是 profiling 看到的是 286 GiB 模型 → 自然得出「6×64 GB 装得下」。

---

## 5. 启动参数逐项解读（`build/compose.yml`）

```yaml
network_mode: host          # 直接占用宿主 8001，省一层 NAT
shm_size: 16gb              # PP worker 多进程共享内存
stop_grace_period: 10m      # 退出要等 6 个 worker 收尾，太短会留僵尸
restart: "no"               # 加载慢，失败要人看日志
```

| 参数 | 值 | 为什么 |
|---|---|---|
| `--pipeline-parallel-size` | 6 | 8 卡里只有 6 卡可用；PP 是唯一能切开 40 层的维度（无 NVLink，TP 跨卡太慢） |
| `--tensor-parallel-size` | 1 | **SSD Engram 硬性要求 TP1**（补丁里显式校验） |
| `VLLM_PP_LAYER_PARTITION` | `7,7,7,7,7,5` | 40 层非均匀切分，末段少 2 层补偿 embedding/lm_head 开销 |
| `--max-model-len` | 1048576 | 1M 上下文（DSV4.1 原生能力） |
| `--max-num-batched-tokens` | 4096 | 与 8 GiB KV 配合，避免 prefill 一次吃爆 |
| `--max-num-seqs` | 32 | 实测 32 是聚合吞吐平台期（767 tok/s） |
| `--kv-cache-dtype` | `fp8_ds_mla` | MLA 结构 fp8 KV，1/2 显存换 2× 容量 |
| `--kv-cache-memory` | 4294967296（4 GiB/rank，共 8 GiB） | ⚠️ 偏小，见踩坑 #6 |
| `--safetensors-load-strategy` | lazy | 必须：SSD 模式要 mmap 原始切片；代价是首次加载极慢 |
| `--engram-config` | `{"cpu_offload":true,"storage":"ssd","engram_cache_gib":8}` | 189 GiB 表落盘 + 8 GiB 热点行缓存 |
| `--speculative-config` | `{"method":"dspark","num_speculative_tokens":5,"use_local_argmax_reduction":true}` | 实测接受 5.4–6.0 token/步 |
| `--disable-custom-all-reduce` | on | 无 NVLink，自定义 all-reduce 更慢 |
| `--enable-prefix-caching` | on | 多轮对话复用前缀，TTFT 从 21 s 降到 220 ms 级 |
| `EAGER=1` | 可选 | 追加 `--enforce-eager`，**首启正确性验证档**；生产勿常开（约 3× 慢） |

**关键环境变量**

| 变量 | 值 | 作用 |
|---|---|---|
| `NVIDIA_VISIBLE_DEVICES` | 0,1,2,3,4,5 | 只用 6 卡，GPU6/7 留给别的活 |
| `NCCL_P2P_DISABLE` / `NCCL_IB_DISABLE` | 1 / 1 | CMP 170HX 不支持 P2P，强开必挂 |
| `NCCL_ALGO` / `NCCL_PROTO` | Ring / Simple | Gen2×4 低带宽下最稳组合 |
| `VLLM_CPU_OFFLOAD_GB_PER_RANK` | `0,0,0,0,0,0` | 显式关掉 vLLM 自带 CPU offload，避免与 Engram SSD 通道抢内存 |
| `VLLM_USE_BREAKABLE_CUDAGRAPH` | 1 | 允许 CUDA graph 被打断（配合投机解码/变长 batch） |
| `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB` | 128 | 稀疏 indexer logits 上限，省显存 |
| `PYTORCH_CUDA_ALLOC_CONF` | expandable_segments:True | 长上下文变长分配，减少碎片 OOM |
| `ulimits.nofile` | 1048576 | AIO + 多 worker 的 fd 需求 |
| `healthcheck.start_period` | **120m** | 冷启动 ~95 分钟，默认 30 s 会一直判不健康 |

---

## 6. Engram SSD 后端工作机制

1. `engram_layer_ids=[1,14]`，两张表各 384 M 行 × head_dim 256（fp8，≈94.4 GiB/表，合计 189 GiB）。
2. 层切分 `7,7,7,7,7,5` → 层 1 落在 **rank0**，层 14 落在 **rank2**；只有这两个 rank 实例化
   `ParallelEngramEmbedding`，其余四个 rank 连 Engram 代码路径都不走。
3. 前向到第 1/14 层时，按 ngram（`max_ngram_size=4`、8 heads）算出需要哪些行号 → 翻译成文件偏移
   （shard 47/48），组织成 4 KB 页、1024 页/批提交 Linux AIO（队列深 128）从 NVMe `O_DIRECT` 直读。
4. 读到的行先进 **8 GiB 有界热点行缓存**（`plan_engram_cache` 把 8 GiB 在两张表间均分 → 每层 4 GiB），
   再 `cudaMemcpyAsync` 到目标 GPU workspace。
5. 缓存淘汰前必须等 CUDA 完成（`dsv41_engram_ssd_clear_cache`），否则会读到被覆盖的行 → 乱码。
6. 读盘完成的同步交给 GPU 侧 kernel：`tl.device_assert(tl.load(status) == 0, "Engram SSD I/O failed")`，
   即 gather kernel 在设备上等 IO 状态标志，不占 CPU 线程、不打断 PP 流水。

**容量账**：8 GiB / 189 GiB ≈ **4.2% 命中率上限**。命中就快，不命中就是一次 NVMe 随机读 + H2D。
这是 c1 只能到 140 tok/s（历史基线 149）、prose 档衰减最明显（每 token 都要查新 ngram）的根本原因。

**PP 与 SSD 的互补点**：PP6 让 6 个 rank 并行推进不同 micro-batch，rank0/rank2 在读盘时其余 rank 仍在算
—— 读盘延迟被流水线掩盖一部分，这是它相对「单卡 TP」的额外收益。

---

## 7. 验收方法

### 7.1 正确性（`build/verify_ssd.py`）

- 12 中文 + 12 英文样本 × repeat，合并检查 `reasoning_content + content`；
- 判 GARBLED：U+FFFD 替换符 / Unicode 私有区码点 / 连续重复退化（同一子串 ≥ N 次）/ 中英混杂乱码；
- **GARBLED 率 = 0% 才算通过**。
- 用法：`python3 verify_ssd.py --base http://127.0.0.1:8001/v1 --model DeepSeek-V4.1-Flash --repeat 2`

### 7.2 吞吐（`bench/benchmark-vllm.mjs`，Node ≥ 18）

```bash
export VLLM_API_KEY=placeholder        # SDK 强制要求，端点本身可无鉴权
node benchmark-vllm.mjs --mode counting --concurrency 1 --max-tokens 500 \
  --base-url http://127.0.0.1:8001 --model DeepSeek-V4.1-Flash \
  --reasoning-effort none --output out.jsonl
```

完整套件 `bench/bench-8001-full.sh`（21 项）：3 模式 × 3 轮 @500 → sweep 1/2/4/8/16/32
→ 3 模式 × 3 轮 @2000 → sweep 1/2/4/8 → **prefill 65536 放最后**（冷缓存下会崩引擎，别丢前面的数据）。

---

## 8. 四轮基准结果（同一脚本同参数，逐步定位瓶颈）

| 轮次 | 实例状态 | counting@500 | code@500 | prose@2000 | 步速 | prefill 65536 | c8 每客户端 |
|---|---|---|---|---|---|---|---|
| 1 | 有 `--enforce-eager`，冷缓存 | 45.2 | 47.1 | 27.4 | 8.5 步/s | 25.76 s / 2084 tok/s | 15.6 |
| 2 | 有 eager，冷缓存 | 45.3 | 47.1 | — | 8.5 步/s | 33.35 s | 15.6（低谷复现） |
| 3 | **无 eager**，刚加载最冷 | 108.1 | 109.2 | 44.9 | 19.5 步/s | **崩（5/6 worker 僵尸）** | — |
| 4 | 无 eager，热缓存 | **140.3** | **120.9** | **77.9** | 23.6 步/s | 21.93 s / 2447 tok/s | **92.2** |
| 历史基线 | 本机旧部署 | 149.1 | 138.4 | 58.8 | — | — | — |

**结论**

1. **`--enforce-eager` 是第一瓶颈**：去掉后 c1 ×2.4–3.1，c8 每客户端 ×6（15.6 → 92.2）。
   根因是 SM80 上 MoE 每层几十个小算子，禁 CUDA graph 后纯 launch-bound。
2. 热缓存后与历史基线差距收敛到 **6–13%** → 剩余瓶颈是 Engram SSD 冷读 + PP6 串行。
3. 并发曲线（第 4 轮 @500）：c1 147.7 → c2 272.7 → c4 481.5 → **c8 754.9（每客 92.2）**
   → c16 716.4（TTFT 834 ms）→ c32 766.8（TTFT 1612 ms）。日常建议 **8–16 并发**。
4. prefill 约 **2.4 K tok/s**（Gen2×4 的 6 卡 PP 是串行瓶颈），长文档首 token 等待明显。

原始数据见 `results/round4-hot.json`。

---

## 9. 踩坑记录（按严重度，对复现者最重要）

### P0

1. **显存装不下（475 GiB vs 384 GB）**：TP 不够、RAID 不成，唯一出路是 Engram 表不进显存。
   为什么选 SSD 而非 RAM：125 GB 内存放不下 189 GiB，且要留给 PP worker。
   **教训**：这类超模型部署，先算「每层参数落在哪」，别先调 vLLM 参数。
2. **官方 vLLM 没有 SSD Engram**：必须从 `344303947/...@d63af5a` 检源码 + 打 38 KB 补丁 + 编译 native `.so`
   覆盖进包目录。基础镜像用 `lazymio/vllm-backport`（pip 装的 vLLM 里没有 `deepseek_v4_1` 目录）。
3. **PP 通信器惰性创建 → profiling 低估显存**：「能起但上负载就 OOM/严重抖动」，很难归因到通信器。
   解法就是 4.1 的 primer 补丁。

### P1

4. **`--enforce-eager` 让性能掉 3 倍**（最贵的一个参数）：45 → 140 tok/s。
   **教训**：eager 只用于首启正确性验证，验证完必须去掉；否则会把「部署上限」误判成「模型不行」。
5. **prefill 65536 打崩引擎**：第 3 轮 5/6 worker 变僵尸（Z 状态）、API 000，但 `/health` 仍 200（**假活**）。
   特征：无 eager + 冷缓存时首次大 prefill 触发一次性巨大激活分配 + 通信器/图捕获抖动。
   规避：**prefill 压测放套件最后**；生产上限单请求长度或先跑小 prefill 预热。第 4 轮（热缓存）未复现。
6. **`engram_cache_gib=8` 太小 → c8 非单调低谷**：8 GiB 只覆盖 4.2% 的表，并发一高行缓存 thrash
   + NVMe 随机读拖住流水线。eager 轮 c4 170 → c8 124.6（每客 15.6）。去 eager 后低谷消失但 c16 仍轻微回落。
   **待办**：调到 16/24/32 GiB 做 A/B（缓存吃内存不吃盘）。
7. **冷启动 ~95 分钟**：lazy 加载 + 489 GB checkpoint + SATA 读 + Engram mmap。
   `start_period: 120m` 才对，默认 30 s 会让编排反复重启、越重启越慢。别用 `restart: always`。

### P2

8. **端口/模型名被外部改动，基准对不上**：原部署在 18060，后来被改成 8000 且 `served-model-name`
   换成别的模型（还去掉了鉴权），再变 8001。压测脚本 `--model` 必须跟着改，否则 404。
   **教训**：压测前先 `GET /v1/models` 对名字，别信昨天的配置。
9. **无 root / 无免密 sudo → 重启依赖机主**：docker 也要 sudo；引擎崩了只能等。归档里的 compose 让机主一条命令重建。
10. **`pkill -f 'bench-8000.sh'` 自杀（踩了两次）**：SSH 远端命令行里含同样字符串 → pkill 匹配到自己的 shell，
   会话被杀（exit 255/137），看起来像「压测把机器搞挂了」。**用 `pgrep` 先确认 PID，再 `kill <pid>`，不要 `pkill -f`。**
11. **`per_request_step_s` 单位误读**：该字段是 **steps/s**，不是毫秒（ms/步 = 1000/步速）。
   曾把 8.5 当 8.5 ms 报出去，结论方向全错。**指标落地前先看脚本里怎么算的。**

### P3

12. **非流式拿不到 reasoning**：`reasoning_content` 非流式恒为空，只能从**流式** `delta.reasoning` 取。
13. **系统盘 92%**：Engram 189 G 在系统 NVMe，只剩 74 G，日志/缓存/镜像层再涨就危险。
    已把容器 `/root/.cache` 外挂；建议把 `/srv/dsv41/engram` 迁到独立 NVMe 或清理旧镜像。
14. **压测端环境**：机器无 node，得自己装（`~/nodejs`；nodejs.org 不通 → npmmirror 源）；
    脚本要 `export VLLM_API_KEY=placeholder`，否则 SDK 直接报错。

---

## 10. 复现步骤（新机器）

```bash
# 1. 准备：6×CMP170HX，/srv/dsv41/{model,engram,cache}，checkpoint 到位
#    model/ 下全部软链到 checkpoint；shard 47/48 单独软链到 /srv/dsv41/engram（NVMe）

# 2. 构建镜像
bash build-port.sh          # 期望末行 BUILD_DONE dsv41-pp6-engram-ssd:port1

# 3. 首启正确性验证（EAGER=1，别省；冷启 ~95 分钟，耐心等 /health）
EAGER=1 docker compose -f build/compose.yml up -d
python3 build/verify_ssd.py --base http://127.0.0.1:8001/v1 \
  --model DeepSeek-V4.1-Flash --repeat 2      # GARBLED 必须 0%

# 4. 生产启动（去掉 eager，重新起来后先跑几个预热请求）
docker compose -f build/compose.yml up -d

# 5. 基准（prefill 放最后）
bash bench/bench-8001-full.sh
```

---

## 11. 与上游的关系（血统）

- **A（并行/模型）**：`344303947/dsv41-flash-pp5-170hx@d63af5a` —— PP5/PP6 层切分、CMP 170HX 的 SM80 适配、
  dspark 投机、1M 上下文、DeepSeek-V4.1 模型实现。
- **B（存储后端）**：`buliaoyin` 谱系的 Engram-SSD 后端（`O_DIRECT + Linux AIO`），移植成补丁。
- **衔接件**：`canoziia` 的 PP communicator primer（3.3 KB），治 A/B 交界的显存时序问题。
- **基础镜像**：`lazymio/vllm-backport@sha256:8094fcbab905…`（`wtdcode/vllm-backport` 的 Docker 镜像）。

三者同源于 `wtdcode/vllm-backport`，所以 `ParallelEngramEmbedding` 的类名、`weight_loader` 协议、
`EngramConfig` 字段两边同名同签名 —— **这是能「补丁叠加」而不是「框架拼接」的前提**。
详细推导见 [fusion.md](fusion.md)。

> ⚠️ **版本锁死**：补丁 `index` 基于 `abc503f..4ef765f` 这对 blob。上游一动 `engram.py` / `gpu_worker.py`，
> 38 KB 补丁就 conflict。所以镜像锁 digest，不跟 `:latest`。
