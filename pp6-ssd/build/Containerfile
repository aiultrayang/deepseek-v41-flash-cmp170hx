# DeepSeek-V4.1-Flash PP6 on 6x CMP 170HX with Engram stored on SSD.
# Stage 1 compiles the ported buliaoyin engram_ssd O_DIRECT backend against
# the CUDA runtime headers shipped inside the SM80 base image; stage 2 is the
# canoziia-style pure-Python vllm overlay plus that .so.
ARG BASE_IMAGE
ARG PKG_DIR=/usr/local/lib/python3.12/dist-packages

FROM ${BASE_IMAGE} AS native
ARG PKG_DIR
USER root
RUN if ! command -v g++ >/dev/null 2>&1; then \
      apt-get update && DEBIAN_FRONTEND=noninteractive \
        apt-get install -y --no-install-recommends g++ && \
      rm -rf /var/lib/apt/lists/*; \
    fi
COPY native/engram_ssd.cpp /tmp/engram_ssd.cpp
RUN set -eux; \
    ARCH=$(uname -m); \
    INC=""; \
    for d in /usr/local/cuda-*/targets/${ARCH}/include /usr/local/cuda-*/include \
             ${PKG_DIR}/nvidia/cu*/include ${PKG_DIR}/nvidia/cuda_runtime/include; do \
      if [ -f "${d}/cuda_runtime_api.h" ]; then INC="${d}"; break; fi; \
    done; \
    test -n "${INC}"; \
    LIB=""; \
    for d in /usr/local/cuda-*/targets/${ARCH}/lib /usr/local/cuda-*/lib64; do \
      if ls "${d}"/libcudart.so >/dev/null 2>&1; then LIB="${d}"; break; fi; \
    done; \
    if [ -z "${LIB}" ]; then \
      for d in /usr/local/cuda-*/targets/${ARCH}/lib ${PKG_DIR}/nvidia/cu*/lib; do \
        if ls "${d}"/libcudart.so.* >/dev/null 2>&1; then LIB="${d}"; break; fi; \
      done; \
    fi; \
    test -n "${LIB}"; \
    echo "using headers ${INC} and lib ${LIB}"; \
    LINK="-L${LIB} -lcudart"; \
    if ! ls "${LIB}"/libcudart.so >/dev/null 2>&1; then \
      SONAME=$(basename "$(ls "${LIB}"/libcudart.so.* | sort | head -1)"); \
      ln -sf "${LIB}/${SONAME}" /tmp/libcudart.so; \
      LINK="-L/tmp -l:libcudart.so -Wl,-rpath,${LIB}"; \
    fi; \
    g++ -O2 -fPIC -std=c++17 -shared \
        -I"${INC}" /tmp/engram_ssd.cpp \
        ${LINK} \
        -Wl,-rpath,"${LIB}" \
        -o /tmp/libdsv41_cuda.so; \
    nm -D /tmp/libdsv41_cuda.so | grep dsv41_engram_ssd_create; \
    ldd /tmp/libdsv41_cuda.so | grep cudart

FROM ${BASE_IMAGE}
ARG PKG_DIR
USER root
COPY vllm/ ${PKG_DIR}/vllm/
COPY --from=native /tmp/libdsv41_cuda.so ${PKG_DIR}/vllm/libdsv41_cuda.so
LABEL io.canozziia.upstream="344303947/dsv41-flash-pp5-170hx@d63af5a" \
      io.workbuddy.patch="prime-pp-communicators + engram-ssd-port"
