#!/bin/bash
# 27B Lorbus INT4 + DFlash (z-lab) draft N=5 — single-stream peak code TPS
# ════════════════════════════════════════════════════════════════════════
#
# Variant 3 (after fp8_e5m2 + TQ_k8v4):
#   - main: Lorbus 27B-int4-AutoRound (same as other variants)
#   - draft: z-lab/Qwen3.6-27B-DFlash (2B BF16 lightweight diffusion drafter)
#   - method=dflash, num_speculative_tokens=5 (per noonghunna recipe; 4 tested but -7% on code workload — 5 wins)
#   - --dtype bfloat16 (workaround vllm#40334 dtype mismatch)
#   - NO --kv-cache-dtype (DFlash needs head_size=256 + non-causal attn,
#     no Ampere backend supports that triple with fp8/turbo KV)
#   - max-num-seqs=1 (single-stream — draft+main eat concurrency budget)
#   - max-model-len 185000 (DFlash adds ~500 MB VRAM)
#
# Expected (noonghunna 2× RTX 3090 quote): 78 narr / 128 code TPS.
# On 2× A5000 should be similar or higher.
set -euo pipefail
docker stop vllm-server-mtp-test 2>/dev/null || true
docker rm vllm-server-mtp-test 2>/dev/null || true

docker run -d \
  --name vllm-server-mtp-test \
  --network genesis-vllm-patches_default \
  --shm-size=8g --memory=64g -p 8000:8000 --gpus all \
  --security-opt label=disable --entrypoint /bin/bash \
  -v /nfs/genesis/models:/models:ro \
  -v /home/sander/.cache/huggingface:/root/.cache/huggingface:ro \
  -v /home/sander/Genesis_Project/vllm_engine/triton-cache-int4-dflash:/root/.triton/cache \
  -v /home/sander/Genesis_Project/vllm_engine/compile-cache-int4-dflash:/root/.cache/vllm/torch_compile_cache \
  -v /home/sander/genesis-vllm-patches/vllm/_genesis:/usr/local/lib/python3.12/dist-packages/vllm/_genesis:ro \
  -v /home/sander/genesis-vllm-patches/tools/genesis_vllm_plugin:/plugin:ro \
  -e VLLM_NO_USAGE_STATS=1 -e VLLM_LOGGING_LEVEL=WARNING \
  -e PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:512" \
  -e VLLM_FLOAT32_MATMUL_PRECISION=high -e VLLM_SSM_CONV_STATE_LAYOUT=DS -e NCCL_P2P_DISABLE=1 -e NCCL_CUMEM_ENABLE=0 \
  -e VLLM_USE_FLASHINFER_SAMPLER=1 -e VLLM_USE_FUSED_MOE_GROUPED_TOPK=1 \
  -e OMP_NUM_THREADS=1 -e CUDA_DEVICE_MAX_CONNECTIONS=8 \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 -e TRITON_CACHE_DIR=/root/.triton/cache \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn -e VLLM_MARLIN_USE_ATOMIC_ADD=1 \
  -e GENESIS_ENABLE_PR40768=1 \
  -e GENESIS_ENABLE_PR40738=1 -e GENESIS_ENABLE_PR40738B=1 \
  \
  -e GENESIS_ENABLE_PR36138=1 \
  -e GENESIS_ENABLE_PR39598=1 \
  \
  -e GENESIS_ENABLE_P68_AUTO_FORCE_TOOL=1 -e GENESIS_ENABLE_P69_LONG_CTX_TOOL_REMINDER=1 \
  -e GENESIS_ENABLE_P72_PROFILE_RUN_CAP=1 -e GENESIS_PROFILE_RUN_CAP_M=4096 \
  -e GENESIS_ENABLE_P74_CHUNK_CLAMP=1 \
  -e GENESIS_ENABLE_PR40941B=1 \
  \
  -e GENESIS_ENABLE_PR41142=1 \
  -e GENESIS_ENABLE_PR34207=1 \
  -e GENESIS_ENABLE_PR41235=1 \
  -e GENESIS_ENABLE_PR40074=1 \
  -e GENESIS_ENABLE_PR41043=1 \
  -e GENESIS_ENABLE_PN17_FA2_LSE_CLAMP=1 \
  -e GENESIS_ENABLE_PN40_DFLASH_OMNIBUS=1 \
  -e GENESIS_ENABLE_PN51_QWEN3_STREAMING_THINKING_DISABLED=0 \
  -e GENESIS_ENABLE_PN50_GDN_FUSED_PROJ=0 \
  -e GENESIS_ENABLE_PR41411=0 \
  -e GENESIS_ENABLE_PN54_GDN_CONTIGUOUS_DEDUP=0 \
  -e GENESIS_ENABLE_PR41602=0 \
  \
  -e GENESIS_ENABLE_PR41418B=0 \
  -e GENESIS_ENABLE_PR40962=0 \
  -e GENESIS_ENABLE_PN59_STREAMING_GDN=0 \
  -e GENESIS_ENABLE_PR41467=0 \
  -e GENESIS_ENABLE_PR41268=1 \
  -e GENESIS_ENABLE_P103=1 \
  \
  -e GENESIS_ENABLE_PR40727=1 \
  -e GENESIS_ENABLE_PR39419=1 \
  -e GENESIS_ENABLE_P78_TOLIST_CAPTURE_GUARD=1 \
  -e GENESIS_PREALLOC_TOKEN_BUDGET=4096 -e GENESIS_BUFFER_MODE=shared \
  -e GENESIS_VLLM_PIN_POLICY=strict \
  vllm/vllm-openai:nightly -c \
  "set -e; echo === 27B Lorbus INT4 + DFlash N=5 z-lab draft ===; \
pip install --quiet --disable-pip-version-check --root-user-action=ignore pandas scipy xxhash; \
cp -r /plugin /tmp/genesis_vllm_plugin; \
pip install --quiet --disable-pip-version-check --root-user-action=ignore --no-deps -e /tmp/genesis_vllm_plugin 2>&1 | tail -3; \
echo === vllm pin ===; pip show vllm 2>/dev/null | head -3; \
python3 -m vllm._genesis.patches.apply_all ; \
exec vllm serve /models/Qwen3.6-27B-int4-AutoRound --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.95 --max-model-len 185000 \
  --max-num-seqs 1 --max-num-batched-tokens 8192 \
  --enable-chunked-prefill --dtype bfloat16 \
  --generation-config vllm \
  --disable-custom-all-reduce --language-model-only --trust-remote-code \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
  --api-key genesis-local --served-model-name qwen3.6-27b \
  --speculative-config '{\"method\":\"dflash\",\"model\":\"/models/Qwen3.6-27B-DFlash\",\"num_speculative_tokens\":5}' \
  --host 0.0.0.0 --port 8000 --disable-log-stats"
sleep 5
docker logs --tail 5 vllm-server-mtp-test 2>&1 | sed 's/^/  /'
echo '27B Lorbus INT4 + DFlash container started.'
