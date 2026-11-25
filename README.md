# DeepSeek-V4.1-Flash on 3× CMP 170HX

This runs DeepSeek-V4.1-Flash-1017 on three NVIDIA CMP 170HX cards, with the
model split between them. It uses vLLM and serves an OpenAI-compatible API that
accepts text and images. The cards don't need a fast link between them; a PCIe
Gen2 x4 connection each is enough.

The checkpoint is
[`codebrandes/DeepSeek-V4.1-Flash-1017-A4NX-2.2bpw`](https://huggingface.co/codebrandes/DeepSeek-V4.1-Flash-1017-A4NX-2.2bpw),
a 2.2-bit EXL3 quantization. The routed experts use the A4NX codebook, which
decodes with integer operations straight into INT8 tensor-core operands, and the
dense weights run from an INT8 copy shipped with the checkpoint. The model's own
MTP layers serve as the drafter.

One user gets around 120–240 tokens per second, depending on what's being
generated. Accuracy is close to the original FP8 release: GSM8K 96.4 % against
97.3 %, MMLU 83.8 % against 85.7 %. The numbers are under
[Benchmarks](#benchmarks).

```bash
git clone https://github.com/AlexBrandes/cmp170hx-deepseek-v41-flash
cd cmp170hx-deepseek-v41-flash
./download.sh     # checkpoint (~167 GB) and DeepSeek's Engram tables (~203 GB) at pinned revisions
./start.sh        # starts the container, waits for /health
```

The API listens on `127.0.0.1:8000` as model `deepseek-v4.1-flash-a4nx-2.2bpw`.
`./stop.sh` stops it. Settings go in `.env` (see `.env.example`).

## Requirements

- 3× NVIDIA CMP 170HX: two with 64 GB and one with 40 GB. List the 40 GB card
  second (`GPUS=0,2,1` if it is device 2). Three 64 GB cards or two 80 GB cards
  also work; see [Other card sets](#other-card-sets). No P2P or link modification needed:
  pipeline parallelism sends one activation per step between the cards.
- Linux x86-64, NVIDIA driver 580 or newer, Docker with the NVIDIA Container Toolkit.
- ~370 GB free on an NVMe SSD: the checkpoint with its INT8 dense weights
  (167 GB) and DeepSeek's Engram tables (203 GB), plus the engine image
  (~8 GB) and compilation caches. The Engram tables are read from disk
  while serving, so keep them on the SSD.
- About 15 GB of host RAM held by the engine for the Engram scales and row cache
  and the drafter's embeddings, on top of the usual vLLM process memory (the
  test host has 93 GB).

The kernels are tuned for this card (sm_80, 74 SMs) and this checkpoint. The
A4NX routed experts only run on this engine.

### Other card sets

The weights take about 149 GiB across the cards and the KV cache is small (about
2.3 KB per token for the whole model), so the cards' memory goes almost entirely
to weights. Only 64/40/64 has been run. These are estimates from the memory the
tested setup reports, not measurements:

| Cards | Layer split | `GPUS` / `.env` | KV cache (estimate) |
|---|---|---|---:|
| 64 + 40 + 64 GB (tested) | 15 / 10 / 15 | `GPUS=0,2,1` | 628,000 tokens (measured) |
| 3 × 64 GB | 13 / 14 / 13 | `LAYER_PARTITION=13,14,13` | about 11 million tokens |
| 2 × 80 GB (not tested) | 20 / 20 | two cards in `GPUS` | about 600,000 tokens |

On two 80 GB cards the model only fits at `GPU_MEMORY_UTILIZATION=0.99`, which
`start.sh` sets when `GPUS` names two cards; at the default 0.97 the engine stops
at startup for lack of memory. That leaves little headroom, and the two-card
layout has not been run.

At 628,000 tokens the cache holds 2.4 requests at the full 262,144-token window;
four sequences of up to about 157,000 tokens each fit at once. With three 64 GB
cards the cache stops being a limit: raise `MAX_NUM_SEQS` in `.env` to use it.

## Benchmarks

Measured on 3× CMP 170HX (64/40/64 GB) at 170 W per card.

### Speed

[sparkDash](https://github.com/MiaAI-Lab/sparkDash) protocol at commit
`3e6d4ef`: greedy, thinking off, output forced to 400 tokens, sent to the engine
endpoint. Each figure is the median of three runs
([results/a4nx-2.2bpw.json](results/a4nx-2.2bpw.json)). The engine serves at most four
sequences; the 8-request level is in the results file.

**Decode, aggregate tok/s**

| Concurrent requests | Kind | tok/s |
| ---: | --- | ---: |
| 1 | code | 200.3 |
| 1 | json | 203.9 |
| 1 | prose | 118.3 |
| 1 | structured | 240.5 |
| 2 | code | 355.7 |
| 2 | json | 343.9 |
| 2 | prose | 212.9 |
| 2 | structured | 333.6 |
| 4 | code | 452.1 |
| 4 | json | 472.7 |
| 4 | prose | 273.9 |
| 4 | structured | 540.3 |

**Prefill** (one cold request)

| Prompt | Tokens | tok/s |
| ---: | ---: | ---: |
| 4k | 4,115 | 3,145 |
| 8k | 8,212 | 3,811 |
| 16k | 16,402 | 4,110 |
| 32k | 32,790 | 4,006 |
| 64k | 65,559 | 3,449 |
| 128k | 131,091 | 2,666 |

**Decode at depth** (our addition, not part of sparkDash: one request after a
long prompt, 400 tokens)

| Context | tok/s |
| ---: | ---: |
| 4k | 115.8 |
| 8k | 116.4 |
| 16k | 116.6 |
| 32k | 113.1 |
| 64k | 113.5 |
| 128k | 102.7 |

### Accuracy

The reference is the original FP8 release (FP8 dense weights, MXFP4 experts,
FP8 KV cache) served by vLLM 0.34.0 on 4× RTX PRO 6000. Both sides were
measured by us with the same prompts and settings.

**Tasks**, sent to the engine directly, with thinking off:

| Check | Original | This |
|---|---:|---:|
| GSM8K, chat, 5-shot (1,319 questions) | 97.3 % | 96.4 % |
| GSM8K, completion, 5-shot (1,319 questions) | 90.8 % | 89.3 % |
| HumanEval+, chat (164 problems) | 87.2 % | 88.4 % |
| IFEval, prompt-level strict (541 prompts) | 88.2 % | 86.9 % |
| MMLU (14,042 questions) | 85.7 % | 83.8 % |
| RULER, 32k context (325 items) | 87.7 % | 84.3 % |
| Image input | | 5/5 |

- **GSM8K** runs vLLM's `tests/evals/gsm8k/gsm8k_eval.py`: 5-shot, the last
  integer in the reply.
- **IFEval** uses Google's instruction checkers.
- **RULER** at 32k: needle retrieval is 96–100 % on both; most of the gap is the
  common-words task, where replies run into the 120-token answer limit.
- **Image input**: a text control, single images (4000×3000 down to 64×64) and
  two images in one message. The model takes no video.
- Each task score is one run; a few questions either way is run-to-run noise.
  MMLU's 1.8-point gap is the quantization's (it is the same before and after
  the kernel work).

**Distance from the original**, teacher-forced on the original's own outputs:
280 chat replies (prose, QA, grounded QA, code, reasoning, multilingual, tool
calls), 565 WikiText passages and long WikiText passages, top-20 logprobs.

| Subset | KL divergence | Same top token | Perplexity ratio |
|---|---:|---:|---:|
| Chat replies | 0.119 | 93.0 % | 1.124 |
| WikiText | 0.207 | 85.4 % | 1.133 |
| Long WikiText | 0.165 | 88.0 % | 1.108 |

## What runs

| | |
|---|---|
| Engine | Three pins from [AlexBrandes/vllm](https://github.com/AlexBrandes/vllm), in `versions.env`: upstream vLLM's wheel for [`a0d7356b98`](https://github.com/vllm-project/vllm/commit/a0d7356b98e929c17051bac44f5a086e4117974d); the fork's DeepSeek-V4.1 vLLM branch at [`a18d750b19`](https://github.com/AlexBrandes/vllm/commit/a18d750b19fd5120554ecf85a9bb1be0058df695), Python files over the wheel; and the vllm-cmp170hx plugin at [`0342f80c51`](https://github.com/AlexBrandes/vllm/commit/0342f80c51fb2837456d231860028880122ee5b6) |
| Image | `ghcr.io/alexbrandes/cmp170hx-dsv41-flash-a4nx@sha256:990be3cf6f219eccdabb3e826253f46a1e06967e327b65e2b00e260813b212eb`: CUDA 13.0 Ubuntu 24.04 base pinned by digest, plus the three pins, each fetched by commit and checked by sha256. `docker/build.sh` rebuilds it (no GPU needed) |
| Checkpoint | [`codebrandes/DeepSeek-V4.1-Flash-1017-A4NX-2.2bpw`](https://huggingface.co/codebrandes/DeepSeek-V4.1-Flash-1017-A4NX-2.2bpw/tree/2aaf191ccb6078319ed41c1bc85ed1b2e6f2c093) at revision `2aaf191ccb6078319ed41c1bc85ed1b2e6f2c093`: routed experts in A4NX at 2.5 bits on layers 0–6 and 36–39 and 2.0 bits on layers 7–35, the MTP layers' experts at 2.0, everything else in EXL3 at 6 bits, the vision tower in BF16. Its model card describes the codebook |
| Dense weights | The checkpoint's `int8/dense.safetensors` (8.1 GB): the attention projections (with the compressor and indexer), the shared experts, the Engram projection, the MTP layers' matrices and the LM head in INT8, made from the original weights with one scale per output row. The dense matrices run as INT8 × INT8 with per-token activation scales; the LM head runs INT8 weights against FP16 activations. The checkpoint's 6-bit copies of these matrices are not loaded |
| Engram tables | DeepSeek's own (shards 47–48 of [`deepseek-ai/DeepSeek-V4.1-Flash-1017`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) at revision `1bae0e1088`), read from the SSD while serving, with scales and an 8 GiB row cache in host memory |
| Drafter | The checkpoint's MTP layers (DSpark), 5 draft tokens per step. The drafter only proposes tokens; it never changes the output |
| Parallelism | pipeline parallel 3, tensor parallel 1, layer split 15/10/15 (`VLLM_PP_LAYER_PARTITION`); the drafter sits on the last card |
| Scheduling | up to 2,048 tokens per step, with up to 512 prompt tokens taken while other requests decode; prefill and decode tokens never share a step, because the tensor-core kernels are chosen per phase |
| Cache | KV records packed to their exact precision (MXFP8 window, NVFP4 compressed, MXFP4 index keys), one block pool per set of pipeline stages; 262,144-token window, up to 4 sequences, GPU memory utilization 0.97 |
| Graphs | `FULL_DECODE_ONLY` CUDA graphs |

[docker/serve.sh](docker/serve.sh) holds every environment variable and engine
argument; `start.sh` mounts it over the copy in the image, so edits take effect
without a rebuild.

The vLLM branch is upstream plus the fork's commits, including DeepSeek-V4.1
hooks and per-stage KV cache pools; the
[plugin](https://github.com/AlexBrandes/vllm/tree/dsv41-flash/plugin) registers the CMP 170HX
(sm_80) platform and carries the kernels. Among them, this model uses:

1. **A4NX routed experts on INT8 tensor cores**, in prefill and decode, with
   nibble decode, fused routing, decode calls of up to 64 rows and a wide
   combine of the expert outputs.
2. **INT8 dense matrices**: INT8 tensor-core GEMMs with per-shape decode
   launch settings, and an INT8-weight Marlin kernel for the LM head in fixed
   64-row calls.
3. **Sparse attention for sm_80**: an MLA-style kernel over packed KV records,
   with the same results whatever the batch (`VLLM_DEEPSEEK_V41_ATTN_MLA=2`).
4. **Fused mHC, router and gate kernels** and exact replacements for the small
   glue operations.
5. **Pipeline scheduling for these kernels**: decodes spread over the in-flight
   micro-batches, one phase per step (`VLLM_PP_SPREAD_DECODES`,
   `VLLM_PP_HOMOGENEOUS_STEPS`).

### Power limit

The recipe leaves the cards' power limit alone. The numbers here were measured
at 170 W per card; to match them, set it yourself
(`sudo nvidia-smi -i <gpu> -pl 170`).

### Reasoning

Thinking is on by default; send `"chat_template_kwargs": {"enable_thinking": false}`
to turn it off. `"reasoning_effort"` in the same object sets how long it thinks:
`low`, `high` (the default), `xhigh`, `max`, or 1–100. The speed tables and task
scores have thinking off.

## Caveats

- One host: 3× CMP 170HX (64/40/64 GB) at PCIe Gen2 x4, Intel Core Ultra 5 225,
  93 GB RAM, Ubuntu 24.04, driver 610.43.02. Other hosts and card sets have not
  been measured.
- This recipe run end to end (`start.sh`, the pinned image and the checkpoint as
  published) gave byte-identical greedy output, top-5 logprobs included, to our
  production deployment of the same model on seven prompts (math, code, prose,
  JSON, French, a 6,400-token lookup, a tool call), and passed the same image
  checks.
- The EXL3 extension and kernels are compiled at build time, so an image rebuilt
  with `docker/build.sh` has its own identity and library hashes.

## Licence

MIT for this repository ([LICENSE](LICENSE)), except `docker/Dockerfile` and
`docker/serve.sh`, which come from the engine fork under Apache-2.0. The engine
is Apache-2.0 and the model has its own licence; see [NOTICE](NOTICE) for every
component and credit.
