# SPDX-License-Identifier: Apache-2.0
# DeepSeek-V4.1-Flash-1017 A4NX 2.2 bpw on three CMP 170HX cards: the engine is three pins.
# Upstream vLLM's prebuilt wheel for the base commit, the fork's DeepSeek-V4.1 vLLM branch (Python files
# only) laid over it, and the vllm-cmp170hx plugin beside it with its native libraries compiled here.
# Building needs network access and CPU/RAM, not a GPU.
FROM nvidia/cuda:13.0.3-devel-ubuntu24.04@sha256:b7ae301dea2c162444795462ce17a05f6a516e5a75944b57af5b88540a1a2266

ARG UPSTREAM_COMMIT=a0d7356b98e929c17051bac44f5a086e4117974d
ARG VLLM_WHEEL=https://wheels.vllm.ai/a0d7356b98e929c17051bac44f5a086e4117974d/vllm-0.33.0rc1.dev659%2Bga0d7356b9-cp38-abi3-manylinux_2_28_x86_64.whl
ARG VLLM_WHEEL_SHA256=5e293d985d0e3964538c989d6123d1b073c2c12d5384c5429e7357868e9e47e2
ARG SOURCE_URL=https://codeload.github.com/AlexBrandes/vllm/tar.gz
ARG VLLM_BRANCH_COMMIT=a18d750b19fd5120554ecf85a9bb1be0058df695
ARG VLLM_BRANCH_SHA256=cb75363fe2bfb107e72166ce3779d9cee3b4b0f8a49a7742f39a28bdd0328e55
ARG PLUGIN_COMMIT=0342f80c51fb2837456d231860028880122ee5b6
ARG PLUGIN_SHA256=6f7b50e7763d2f81db5a982f8d300306c25a95131578ff6b975cdd5485e071e7

ENV DEBIAN_FRONTEND=noninteractive \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:${PATH}
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       python3.12 python3.12-venv python3.12-dev ca-certificates curl ninja-build \
    && rm -rf /var/lib/apt/lists/* \
    && python3.12 -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir uv==0.9.8

# 1. Upstream vLLM at the base commit, with its pinned dependencies.
RUN curl -fsSL -o /tmp/vllm.whl "${VLLM_WHEEL}" \
    && echo "${VLLM_WHEEL_SHA256}  /tmp/vllm.whl" | sha256sum -c - \
    && mv /tmp/vllm.whl "/tmp/$(basename "${VLLM_WHEEL}" | sed 's/%2B/+/')" \
    && uv pip install --no-cache /tmp/vllm-*.whl \
    && rm -f /tmp/vllm-*.whl

# 2. The vLLM branch (upstream + the fork's commits, including the DeepSeek-V4.1 hooks and the per-stage KV
#    cache pools): its Python files over the installed package. The wheel keeps its compiled extensions.
RUN test -n "${VLLM_BRANCH_COMMIT}" && test -n "${VLLM_BRANCH_SHA256}" \
    && curl -fsSL -o /tmp/vllm-branch.tar.gz "${SOURCE_URL}/${VLLM_BRANCH_COMMIT}" \
    && echo "${VLLM_BRANCH_SHA256}  /tmp/vllm-branch.tar.gz" | sha256sum -c - \
    && mkdir /tmp/vllm-branch \
    && tar -xzf /tmp/vllm-branch.tar.gz --strip-components=1 -C /tmp/vllm-branch \
    && SP=$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])') \
    && cd /tmp/vllm-branch \
    && find vllm -type f -name '*.py' ! -name '_version.py' -exec cp --parents '{}' "$SP/" \; \
    && rm -rf /tmp/vllm-branch /tmp/vllm-branch.tar.gz

# 3. The plugin, with the EXL3 extension (_exl3_C) built against this image's PyTorch.
RUN test -n "${PLUGIN_COMMIT}" && test -n "${PLUGIN_SHA256}" \
    && curl -fsSL -o /tmp/plugin.tar.gz "${SOURCE_URL}/${PLUGIN_COMMIT}" \
    && echo "${PLUGIN_SHA256}  /tmp/plugin.tar.gz" | sha256sum -c - \
    && mkdir /opt/vllm-cmp170hx \
    && tar -xzf /tmp/plugin.tar.gz --strip-components=1 -C /opt/vllm-cmp170hx \
    && rm /tmp/plugin.tar.gz \
    && cd /opt/vllm-cmp170hx \
    && MAX_JOBS=4 TORCH_CUDA_ARCH_LIST=8.0 python csrc/exl3/build_standalone.py --out vllm_cmp170hx \
    && uv pip install --no-cache --no-deps . \
    && python -c "import importlib.metadata as m; \
assert [e.value for e in m.entry_points(group='vllm.general_plugins') if e.name == 'cmp170hx'] == ['vllm_cmp170hx.plugin:register']; \
assert [e.value for e in m.entry_points(group='vllm.platform_plugins') if e.name == 'cmp170hx'] == ['vllm_cmp170hx.plugin:platform_plugin']"

# 4. Native libraries the plugin loads by path: the INT8 kernels for the dense sidecar (rotation, INT8 activation
#    quantization, W8 decode GEMMs; csrc/exl3/lab_once), a 128x256x64 INT8 CUTLASS GEMM (CUTLASS headers from the
#    vLLM wheel), and the sparse MLA attention kernel for DeepSeek-V4.1 (csrc/dsv41_mla).
RUN cd /opt/vllm-cmp170hx/csrc && mkdir -p /opt/exl3-once /opt/dsv41-mla \
    && SP=$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])') \
    && TI=$SP/torch/include && CUT=$SP/vllm/third_party/fmha_sm100/cutlass \
    && ABI=$(python -c 'import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))') \
    && F="-O3 -std=c++20 -gencode=arch=compute_80,code=sm_80 --expt-relaxed-constexpr -DNDEBUG -Xcompiler -fPIC -shared -isystem $TI -isystem $TI/torch/csrc/api/include -D_GLIBCXX_USE_CXX11_ABI=$ABI -L$SP/torch/lib -lc10 -lc10_cuda -ltorch_cpu -ltorch_cuda -ltorch" \
    && (cd exl3 && nvcc $F -I. lab_once/exl3_i8_lab.cu -o /opt/exl3-once/exl3_i8_lab.so) \
    && (cd exl3 && nvcc $F -I$CUT/include -I$CUT/tools/util/include lab_once/cmp_i8mm.cu -o /opt/exl3-once/cmp_i8mm.so) \
    && nvcc $F dsv41_mla/sparse_mla_sm80.cu -o /opt/dsv41-mla/sparse_mla.so

COPY serve.sh /opt/cmp170hx/serve.sh
LABEL org.opencontainers.image.source="https://github.com/AlexBrandes/vllm" \
      org.opencontainers.image.title="DeepSeek-V4.1-Flash-1017 A4NX PP3 on CMP 170HX" \
      cmp170hx.upstream_commit="${UPSTREAM_COMMIT}" \
      cmp170hx.vllm_commit="${VLLM_BRANCH_COMMIT}" \
      cmp170hx.plugin_commit="${PLUGIN_COMMIT}"
WORKDIR /cache
ENTRYPOINT ["/bin/bash", "/opt/cmp170hx/serve.sh"]
