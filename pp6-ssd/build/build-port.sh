#!/usr/bin/env bash
# 组装并构建移植镜像：344303947/dsv41-flash-pp5-170hx@d63af5a + canoziia primer + engram-ssd port
# 用法：把本仓库 build/ patches/ native/ 放到 $R（默认 /srv/dsv41/port），然后 bash build-port.sh
set -euox pipefail
R=/srv/dsv41/port
BASE_IMAGE=docker.io/lazymio/vllm-backport@sha256:8094fcbab905a04a480b327f2761255e3d17cd8d39470cac9d76450bbb567f7e
OUT_IMAGE=dsv41-pp6-engram-ssd:port1

cd "$R"
# 1. 源码树（可重复执行）
if [[ ! -d src/.git ]]; then
  git init -q src
  git -C src remote add origin https://github.com/344303947/dsv41-flash-pp5-170hx.git
fi
cd src
git fetch --depth 1 origin d63af5a472dc76b12d7a73d50a5af142844c15d1
git checkout -q --detach FETCH_HEAD
git clean -qfd
git config user.email x@x; git config user.name x
# 2. canoziia primer + 移植补丁
git apply --check ../0001-prime-pp-communicators-before-model-load.patch
git apply ../0001-prime-pp-communicators-before-model-load.patch
git apply --check ../engram-ssd-port.patch
git apply ../engram-ssd-port.patch
python3 -m py_compile \
  vllm/models/deepseek_v4_1/common/engram.py \
  vllm/models/deepseek_v4_1/common/engram_ssd.py \
  vllm/models/deepseek_v4_1/host_memory.py \
  vllm/config/engram.py vllm/config/vllm.py \
  vllm/v1/worker/gpu_worker.py
grep -q "_pp_communicators_primed" vllm/v1/worker/gpu_worker.py
cd "$R"
# 3. 构建上下文
rm -rf context
mkdir -p context/native
cp -a src/vllm context/vllm
cp src/csrc/cpu/dsv41/engram_ssd.cpp context/native/
# 确认基础镜像的 python 包目录
PKG_DIR=$(docker run --rm --entrypoint python3 "$BASE_IMAGE" -c \
  "import sysconfig;print(sysconfig.get_paths()['purelib'])")
echo "base package dir: $PKG_DIR"
# 4. 镜像构建（native .so + python overlay）
DOCKER_BUILDKIT=1 docker build -t "$OUT_IMAGE" \
  --build-arg BASE_IMAGE="$BASE_IMAGE" --build-arg PKG_DIR="$PKG_DIR" \
  -f Containerfile context
docker run --rm --entrypoint bash "$OUT_IMAGE" -c \
  "ls -la \$(python3 -c \"import vllm,os;print(os.path.dirname(vllm.__file__))\")/libdsv41_cuda.so && \
   python3 -c 'import ctypes; l=ctypes.CDLL(\"libdsv41_cuda.so\"); print(\"native version:\", l.dsv41_engram_ssd_version())'" 2>/dev/null || true
echo "BUILD_DONE $OUT_IMAGE"
df -h /
