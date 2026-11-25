#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# DeepSeek-V4.1-Flash-1017, A4NX routed experts (2.2 bpw average), pipeline-parallel on CMP 170HX cards, DSpark
# drafter (the checkpoint's own MTP layers).
# /model:  codebrandes/DeepSeek-V4.1-Flash-1017-A4NX-2.2bpw (includes the INT8 dense weights in int8/)
# /engram: the Engram tables from deepseek-ai/DeepSeek-V4.1-Flash-1017 (model.safetensors.index.json and
#          shards 47-48), read from disk at run time
# Card count and layer split come from start.sh: PP_SIZE, LAYER_PARTITION, GPU_MEMORY_UTILIZATION.
set -euo pipefail

export VLLM_CACHE_ROOT=/cache/vllm
export TRITON_CACHE_DIR=/cache/triton
export CUDA_CACHE_PATH=/cache/cuda
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PP_SIZE=${PP_SIZE:-3}
# Layers per card, in card order (the 40 GB card in the middle on 64/40/64).
export VLLM_PP_LAYER_PARTITION=${LAYER_PARTITION:-15,10,15}
# Scheduling: spread decodes over the in-flight micro-batches and never mix prefill and decode tokens in one step.
export VLLM_PP_SPREAD_DECODES=1
export VLLM_PP_HOMOGENEOUS_STEPS=1

# Routed experts (A4NX): INT8 tensor-core GEMMs with per-row INT8 activations in prefill and decode, nibble decode,
# fused routing, decode calls up to 64 rows and the wide combine.
export VLLM_EXL3_MOE_PREFILL_TC=1
export VLLM_EXL3_MOE_DECODE_TC=1
export VLLM_EXL3_MOE_DECODE_TC_MODE=full
export VLLM_EXL3_MOE_DECODE_TC_MAX_ROWS=64
export VLLM_EXL3_Q4N_PREFILL=i8
export VLLM_EXL3_Q4N_I8_GEMM=t2c4
export VLLM_EXL3_Q4N_DECODE=i8
export VLLM_EXL3_Q4N_DECODE_GEMM=t2c4
export VLLM_EXL3_Q4N_DECODE_NIBBLE=1
export VLLM_EXL3_MOE_ROUTE_FUSE=1
export VLLM_EXL3_MOE_COMBINE_WIDE=1
export VLLM_EXL3_ROUTE_CHECK=0
export VLLM_EXL3_ACT_REG=1
# EXL3 tensor-core paths for any dense matrix not served from the INT8 file.
export VLLM_EXL3_DENSE_PREFILL_TC=1
export VLLM_EXL3_DENSE_DECODE_TC=1
export VLLM_EXL3_DENSE_DECODE_TC_FAMILIES=all
export VLLM_EXL3_DENSE_DECODE_SPLITK=1
export VLLM_EXL3_DECODE_VARIANT=off
# Dense matrices from the checkpoint's INT8 copy: INT8 tensor-core GEMMs (per-shape decode launch table below,
# grouped activation quantization for the grouped output projection), the LM head as fixed 64-row Marlin W8A16 calls.
export VLLM_EXL3_DENSE_W8=/model/int8/dense.safetensors
export VLLM_EXL3_DENSE_W8_DECODE=1
export VLLM_EXL3_DENSE_W8_GROUPED_ACT=1
export VLLM_EXL3_DENSE_W8_LM_HEAD_ROWS=64
export VLLM_EXL3_W8_TUNE=32768:1280:1:2:2,32768:1280:2:2:4,5120:8192:2:4:1,4608:5120:1:8:2,4608:5120:2:4:4,5120:2304:1:4:4,5120:2304:2:8:1,1280:5120:1:8:4,1280:5120:2:8:1,512:5120:1:8:4,512:5120:2:4:4,1792:5120:1:8:4,1792:5120:2:4:4,g:1024:4096:1:4:1,g:1024:4096:2:4:1
# DeepSeek-V4.1 sm_80 kernels: MLA-style sparse attention (batch invariant), packed KV records (MXFP8 window,
# NVFP4 compressed, MXFP4 index keys), fused mHC, router and gate GEMV, exact glue ops, in-place RoPE.
export VLLM_DEEPSEEK_V41_ATTN_MLA=2
export VLLM_DEEPSEEK_V41_KV_PACK=1
export VLLM_DEEPSEEK_V41_SWA_META_BATCH=1
export VLLM_DEEPSEEK_V41_TRIM_ROPE=1
export VLLM_DEEPSEEK_V41_ROPE_INPLACE=1
export VLLM_DEEPSEEK_V41_RMS_ANY=1
export VLLM_DEEPSEEK_V41_FUSED_MHC=1
export VLLM_DEEPSEEK_V41_FUSED_MHC_FINISH=1
export VLLM_DEEPSEEK_V41_FUSED_MHC_FINISH_V2=1
export VLLM_SM80_MHC_UNIFORM_KERNEL_A=1
export VLLM_DEEPSEEK_V41_GATE_GEMV=1
export VLLM_DEEPSEEK_V41_GATE_GEMV_MAX_ROWS=2048
export VLLM_DEEPSEEK_V41_GATE_GEMV_TUNE=1
export VLLM_DEEPSEEK_V41_GATE_FP16=1
export VLLM_DEEPSEEK_V41_FUSED_ROUTE=1
export VLLM_DEEPSEEK_V41_SELECT_ONE_LAUNCH=1
export VLLM_DEEPSEEK_V41_EXACT_GLUE=1
export VLLM_SHARED_EXPERT_REORDER=1
# Token embeddings in INT8; the drafter's embedding rows in pinned host memory.
export VLLM_DEEPSEEK_V41_EMBED_INT8=1
export VLLM_DEEPSEEK_V41_DSPARK_EMBED_INT8=1
export VLLM_DEEPSEEK_V41_DSPARK_EMBED_HOST=1
# Engram tables stay on disk: scales and an 8 GiB row cache in host memory, other rows read on demand.
export VLLM_ENGRAM_PLACEMENT=ssd
export VLLM_ENGRAM_ROW_CACHE_GIB=8

exec vllm serve /model \
  --host 0.0.0.0 --port 8000 \
  --served-model-name deepseek-v4.1-flash-a4nx-2.2bpw \
  --quantization exl3 --load-format exl3 --dtype bfloat16 --tokenizer-mode deepseek_v41 \
  --pipeline-parallel-size "$PP_SIZE" --tensor-parallel-size 1 --mm-encoder-tp-mode data \
  --max-model-len "${MAX_MODEL_LEN:-262144}" --kv-cache-pool-per-pp-stage \
  --max-num-seqs "${MAX_NUM_SEQS:-4}" --max-num-batched-tokens 2048 --prefill-chunk-with-decodes 512 \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.97}" \
  --enable-auto-tool-choice --tool-call-parser deepseek_v41 --reasoning-parser deepseek_v41 \
  --enable-prompt-tokens-details \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
  --hf-overrides '{"engram_table_dir": "/engram"}' \
  --speculative-config '{"method": "dspark"}' \
  "$@"
