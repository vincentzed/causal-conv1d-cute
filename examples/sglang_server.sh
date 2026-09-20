#!/usr/bin/env bash
# Serve a Gated DeltaNet model with sglang and let causal-conv1d-cute handle its conv calls.
#
# sglang discovers the plugin through the "sglang.srt.plugins" entry point that this package
# installs, and loads it in every scheduler process. The plugin registers nothing unless
# CAUSAL_CONV1D_CUTE_SGLANG=1, so installing the package never changes a server by itself.
#
# The flags below are the sglang cookbook recipe for Qwen3.8-27B with its in-checkpoint MTP head
# (docs/cookbook/autoregressive/Qwen/Qwen3.8-27B.mdx, single GPU). With MTP the target model verifies
# four draft tokens per step, which reaches the multi-token causal_conv1d_update path; prefill
# reaches causal_conv1d_fn. Without the four --speculative flags, decode reaches sglang's fused
# projection unpack and conv call instead, which the plugin serves as well.
#
# On startup the scheduler log shows which calls the kernels serve and, for calls that are passed
# on to sglang's own op, the reason:
#
#   causal_conv1d_update served: input (48, 10240, 4) strides (40960, 1, 10240) torch.bfloat16
#   causal_conv1d_fn served: input (10240, 2048) strides (1, 16384) torch.bfloat16
#   causal_conv1d_fn passed on: input (10240, 12) ... (12 tokens is fewer than 16)
#   causal_conv1d_unpack served: input (115, 16384) strides (16384, 1) torch.bfloat16   (no MTP)
set -euo pipefail

pip install causal-conv1d-cute

CAUSAL_CONV1D_CUTE_SGLANG=1 sglang serve \
  --trust-remote-code \
  --model-path Qwen/Qwen3.8-27B \
  --kv-cache-dtype fp8_e4m3 \
  --mem-fraction-static 0.85 \
  --chunked-prefill-size 2048 \
  --mamba-radix-cache-strategy extra_buffer \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --host 127.0.0.1 --port 30000
