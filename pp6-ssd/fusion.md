# 三个开源件是怎么融合成这套 PP6 + Engram-SSD 部署的

> 配套代码：`build/Containerfile`、`build/build-port.sh`、`patches/*.patch`、`native/engram_ssd.cpp`
> 与 [README.md](README.md) 配套阅读：README 讲「怎么搭」，本文讲「为什么能这么搭」。

---

## 0. 先认清"两个项目"是谁

|       | 项目                                          | 提交 / 来源                                                            | 负责什么                                                                                |
| ----- | ------------------------------------------- | ------------------------------------------------------------------ | ----------------------------------------------------------------------------------- |
| **A** | `344303947/dsv41-flash-pp5-170hx`           | `@d63af5a472dc76b12d7a73d50a5af142844c15d1`（2026-09-14 建，10-19 更新） | PP5/PP6 层切分、CMP 170HX 的 SM80 适配、dspark 投机解码、1M 上下文、DeepSeek-V4.1 模型实现               |
| **B** | `buliaoyin` 的 Engram-SSD 后端                 | 由 `buliaoyin/vllm-backport` 谱系移植                                   | `O_DIRECT + Linux AIO` 直读 checkpoint 切片、有界热点行缓存（`engram_ssd.cpp` / `engram_ssd.py`） |
| （基础）  | `lazymio/vllm-backport@sha256:8094fcbab905` | `wtdcode/vllm-backport` 的 Docker 镜像                                | 提供 vLLM 0.13.1 + torch 2.13.0+cu130 + SM80 kernel 基线                                |
| （衔接件） | `canoziia` 的 PP communicator primer         | 3.3 KB patch                                                       | PP 通信器预热，解决 A/B 交界处的显存时序                                                            |

### 为什么"能融合"的前提是血统

`buliaoyin/vllm-backport` 与 `344303947/dsv41-flash-pp5-170hx` 都是 `wtdcode/vllm-backport`（"Dogfooding vLLM backport for older GPUs"）的下游分叉，基础镜像同为 `lazymio/vllm-backport`。

因此：

- `ParallelEngramEmbedding` 的类名、`weight_loader` 协议、`EngramConfig` 字段在两边**同名同签名**；
- B 的 38 KB patch 才能 `git apply` 干净落地（`index abc503f..4ef765f`）；
- 换成两个无关框架（比如换 SGLang 或另一个 vLLM 大版本），就只能重写，补丁方式失效。

**一句话：不是两个框架拼接，而是同一棵树的两个分叉，用补丁叠加。**

---

## 1. 为什么非融不可（动机账）

A 的 README 明确写着：

> `dsv41-flash-pp5-170hx & pp6` Supports the original (full/unquantized) model. Prefill 7,000 t/s, decode 45-75-125 t/s. dspark. **Requires 200 GB of memory.** Supports 1M context.

也就是说 A 的默认姿态是「Engram 走 CPU offload 放内存」，要 200 GB RAM。

八卡机现实：**RAM 只有 125 GB**，而 Engram 两表合计 189 GiB（各 384 M 行 × head_dim 256，FP8）。

于是换掉且只换掉 A 里"Engram 放哪"这一件事 —— 用 B 的 SSD 后端把 189 GiB 挪到 NVMe，内存需求从 200 GB 掉到几 GB：

```
A 原方案：  GPU 384 GB + RAM 200 GB(Engram)   → 这台机器 RAM 不够 ✗
融合方案：  GPU 384 GB + NVMe 189 GB(Engram)   → 成立 ✓
```

这也是启动命令里 `VLLM_CPU_OFFLOAD_GB_PER_RANK=0,0,0,0,0,0`（显式关掉 vLLM 自带 CPU offload）的原因：Engram 的 offload 已由 B 接管，不能再让 vLLM 自己的通道分一份。

**A 的并行方案（PP6）完全没动** —— 这是理解融合的关键：B 只换存储后端，不碰并行。

---

## 2. 融合的六个接缝

### 接缝①　代码层：补丁叠加（不是 merge 两个 fork）

```
A 的源码树 (git checkout d63af5a)
   ├─ git apply 0001-prime-pp-communicators-before-model-load.patch   (3.3 KB, 1 文件)
   └─ git apply engram-ssd-port.patch                                 (38 KB, 6 文件)
        ├─ 新增 csrc/cpu/dsv41/engram_ssd.cpp      ← B 的 C++ 后端
        ├─ 新增 vllm/models/deepseek_v4_1/common/engram_ssd.py
        │         （Python 侧：表几何、行号→文件偏移、CUDA 完成后再清缓存）
        ├─ 改   vllm/models/deepseek_v4_1/common/engram.py    （ram/ssd 分流）
        ├─ 改   vllm/models/deepseek_v4_1/host_memory.py      （SSD 模式不再分配 189 GiB 主机内存）
        ├─ 改   vllm/config/engram.py                          （storage 枚举 + cache_gib + 校验）
        └─ 改   vllm/config/vllm.py                            （把 engram 配置接到并行/加载策略）
```

打补丁前后各有守门检查（`build-port.sh`）：

```bash
git apply --check ../0001-...patch && git apply ../0001-...patch
git apply --check ../engram-ssd-port.patch && git apply ../engram-ssd-port.patch
python3 -m py_compile vllm/models/deepseek_v4_1/common/engram.py \
        vllm/models/deepseek_v4_1/common/engram_ssd.py vllm/models/deepseek_v4_1/host_memory.py \
        vllm/config/engram.py vllm/config/vllm.py vllm/v1/worker/gpu_worker.py
grep -q "_pp_communicators_primed" vllm/v1/worker/gpu_worker.py
```

### 接缝②　构建层：两阶段镜像，A 不进镜像、B 只贡献一个 .so

```dockerfile
FROM ${BASE_IMAGE} AS native
  # 装 g++，自动探测 cuda_runtime_api.h / libcudart.so 的真实位置
  # （系统 CUDA、pip 的 nvidia/cu*/include 两种布局都兼容）
  g++ -O2 -fPIC -std=c++17 -shared -I${INC} engram_ssd.cpp -L${LIB} -lcudart -o libdsv41_cuda.so
  nm -D libdsv41_cuda.so | grep dsv41_engram_ssd_create     # 验导出符号
  ldd  libdsv41_cuda.so | grep cudart                       # 验动态链接

FROM ${BASE_IMAGE}
  COPY vllm/ ${PKG_DIR}/vllm/                     # 只 overlay Python 树
  COPY --from=native /tmp/libdsv41_cuda.so ${PKG_DIR}/vllm/libdsv41_cuda.so
LABEL io.canozziia.upstream="344303947/dsv41-flash-pp5-170hx@d63af5a"
LABEL io.workbuddy.patch="prime-pp-communicators + engram-ssd-port"
```

结果：镜像 29.7 GB 主要是基础镜像，A 的 231 MB 源码树整体不进层；C++ 与 Python 之间用 **C ABI**（`dsv41_engram_ssd_{create,enqueue,stats,clear_cache,destroy,version}`）隔离，避免 A/B 的 Python 侧版本细节互相污染。

### 接缝③　并行约束：B 要求 TP1，A 用 PP —— 两条路只剩一条

`engram.py` 里显式拒绝：

```python
if storage == "ssd" and tp_size != 1:
    raise ValueError("Engram SSD currently requires TP1")
```

而 A 的 PP6 天然是 `--pipeline-parallel-size=6 --tensor-parallel-size=1`。

**所以"PP6 + SSD"不是凑巧凑出来的组合，是约束唯一解**：B 排除了 TP，A 排除了"装不下"，交集只剩 PP。

### 接缝④　显存账：把 189 GiB 变成 `meta`，让 profiling 看不见

这是整套方案真正让 384 GB 够用的一招，全在 `ParallelEngramEmbedding.__init__` 里：

```python
if storage == "ssd":
    self.ssd = EngramSSD(dim, block_size, self.part_n_hash_cols, max_tokens,
                         self.vocab_start_idx, self.vocab_end_idx, cache_bytes)

    # 保留 checkpoint 的名字和形状，但不分配表本身
    def _meta_alloc(shape, dtype):
        return torch.empty(shape, dtype=dtype, device="meta")
    alloc = _meta_alloc
else:
    alloc = _alloc

self.weight = nn.Parameter(alloc((part_num_embeddings, dim), torch.float8_e4m3fn), requires_grad=False)
self.weight_scale_inv = nn.Parameter(alloc((part_num_embeddings, dim // 32), torch.uint8), requires_grad=False)

for p in (self.weight, self.weight_scale_inv):
    set_weight_attrs(p, {"weight_loader": self.ssd.load if self.ssd else _engram_head_shard_weight_loader,
                         "engram_vocab_start": self.vocab_start_idx})
```

- `device="meta"` = 零字节分配；
- `weight_loader` 换成 `ssd.load` = **只登记"文件哪个偏移、多少行"，不拷贝数据**。

于是 A 的显存 profiling 看到的是 286 GiB 模型，自然算出"6×64 GB 装得下"，KV cache 还能留 8 GiB。

### 接缝⑤　执行层：层归属决定谁读盘，读盘挂在 PP 流水的层前向上

`config.json` 里 `engram_layer_ids = [1, 14]`，A 的层切分 `VLLM_PP_LAYER_PARTITION=7,7,7,7,7,5` 推出：

| rank  | 层区间   | 有无 Engram                        |
| ----- | ----- | -------------------------------- |
| rank0 | 0–6   | **层 1 → 有**（AIO 线程 + 4 GiB 行缓存）  |
| rank1 | 7–13  | 无                                |
| rank2 | 14–20 | **层 14 → 有**（AIO 线程 + 4 GiB 行缓存） |
| rank3 | 21–27 | 无                                |
| rank4 | 28–34 | 无                                |
| rank5 | 35–39 | 无                                |

- 只有 rank0 / rank2 实例化 `ParallelEngramEmbedding`，另外四个 rank 连 Engram 代码路径都不走；
- 缓存预算：`plan_engram_cache(8 GiB, (384006168, 384016682), 256)` 按 FP8 字节数在两表间**均分并互让余量** → 每层 4 GiB 主机 pinned 缓存；
- 前向到第 1/14 层时，Python 算出要哪些行 → 组织 4 KB 页（1024 页/批）提交 Linux AIO（队列深 128）→ 命中缓存就直接拷，未命中就 `O_DIRECT` 直读 NVMe；
- 读盘完成的同步交给 GPU 侧 kernel：`tl.device_assert(tl.load(status) == 0, "Engram SSD I/O failed")`，即 gather kernel 在设备上等 IO 状态标志，不占 CPU 线程、不打断 PP 流水。

**这正是两个方案互补的地方**：A 的 PP6 让 6 个 rank 并行推进不同 micro-batch，rank0 在读盘时其它 rank 仍在算 —— 读盘延迟被流水线掩盖一部分，这是它相对"单卡 TP"的额外收益。

### 接缝⑥　数据层：同一棵 checkpoint 树，两条盘路

```
/srv/dsv41/model/DeepSeek-V4.1-Flash/
  ├─ model-00001…00046-of-00048.safetensors → <CKPT_ROOT>/…                (SATA 2T，只读一次)
  ├─ model-00047-of-00048.safetensors       → /srv/dsv41/engram/…           (NVMe，随机读主力)
  ├─ model-00048-of-00048.safetensors       → /srv/dsv41/engram/…
  └─ config.json / tokenizer …              → <CKPT_ROOT>/DeepSeek-V4.1-Flash-1017/
```

- A 的 `--safetensors-load-strategy=lazy`（B 强制 `load_config.safetensors_load_strategy="lazy"`，否则报错）；
- B 直读**原始 checkpoint 切片**，不做 ETL 转格式 → 省掉一次 189 GiB 的格式转换；
- 两条 I/O 走不同物理盘，互相不抢带宽。

---

## 3. 衔接件：为什么必须再加 canoziia 的 primer

B 释放了 189 GiB 显存，看似宽裕；但 vLLM 的 PP 点对点通信器是**惰性创建**（首次 send/recv 才分配 NCCL 通信器，5 对 rank × 2 方向），而显存 profiling 发生在模型加载后、推理前。

如果不预热：profiling 时通信器还没建 → 账面够用 → 上负载首次建通信器 → OOM 或剧烈抖动，表现为"能起来但一压就炸"，极难归因。

补丁做法（`gpu_worker.py`）：

```python
self.model_runner = GPUModelRunnerV1(self.vllm_config, self.device)
self._prime_pp_communicators()      # ← 在模型/KV 分配之前
```

`_prime_pp_communicators()` 用 `int32[1]` 的 primer tensor 走一遍每对 rank 的 send/recv，把通信器提前建好，`world_size<=1` 直接短路，`_pp_communicators_primed` 保证幂等。

**它是 A/B 交界处的产物**：A 的 PP6 需要通信器，B 的显存释放让这些通信器的开销变得"看得见、必须提前算"，两者共同逼出这个补丁。

---

## 4. 融合后的运行时形态

```
docker run dsv41-pp6-engram-ssd:port1
  └─ vllm serve（1 个 API 进程 + 6 个 worker 进程，spawn）
       rank0 ── 层0-6  ── Engram L1  → AIO 读 NVMe
       rank1 ── 层7-13
       rank2 ── 层14-20 ── Engram L14 → AIO 读 NVMe
       rank3 ── 层21-27
       rank4 ── 层28-34
       rank5 ── 层35-39
       通信：NCCL Ring / Proto=Simple / P2P 关（CMP 170HX 无 NVLink、PCIe Gen2×4）
```

---

## 5. 融合的代价与风险

| 风险                  | 说明                                                                                                                                                     |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **版本锁死**            | 补丁 `index` 基于 `abc503f..4ef765f` 这对 blob。上游一动 `engram.py` / `gpu_worker.py`，38 KB 补丁就 conflict，必须重新对位。所以镜像锁 digest `@sha256:8094fcbab905`，不跟 `:latest` |
| **两个 fork 漂移**      | A 若再更新 PP 逻辑，B 的 `engram_ssd.py` 调用的 `ParallelEngramEmbedding` 内部结构可能变（现在依赖 `part_n_hash_cols`、`vocab_start_idx`、`_views`）                             |
| **TP 永不可用**         | B 的 TP1 是硬校验，除非重写 SSD 侧分片逻辑，否则这台机器永远只能 PP                                                                                                              |
| **单点：两个 rank 承担读盘** | 层 1/14 都在中层，rank0（首段）和 rank2 各背一份读盘压力；若把 engram 层切到同一 rank，缓存会争抢（当前分布已经是较优解）                                                                           |
| **诊断困难**            | C ABI 把错误压成 `dsv41_engram_ssd_error()` 字符串 + kernel 里的 `device_assert`，出错表现为"输出乱码"或"gather 报 I/O failed"，必须靠 `verify_ssd.py` 的乱码判据把关                   |



---

## 6. 一句话结论

> **A 出的是"怎么把 40 层塞进 6 张卡并行跑"（PP6 + dspark + SM80），B 出的是"怎么让 189 GiB Engram 不住显存也不住内存"（O_DIRECT 直读 + 行缓存）。两者靠"层归属 → 谁读盘"这一根线缝在一起：B 只把 Engram 层的权重换成 `meta` 占位、把加载器换成 `ssd.load`，A 的并行与显存账完全不用改；再加上 canoziia 的 PP 通信器预热，把 B 腾出来的显存提前算清，才真正跑到 140 tok/s。**

---

相关文件：`README.md`（完整部署方案与 14 条踩坑）、`build/build-port.sh`、`patches/engram-ssd-port.patch`、`native/engram_ssd.cpp`
