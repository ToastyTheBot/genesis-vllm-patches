#!/bin/bash
# Phase 3.5 — v789: v775 baseline + GENESIS_ENABLE_PR40849=1 only
#
# 35B-A3B-FP8 = online-quantized target with MTP draft.
# PR40849 propagates online-quant config to MTP draft worker → frees ~600 MiB
# in draft. Should be no-op on TPS but VRAM headroom unlocks longer ctx
# OR more max_num_seqs.
#
# Win condition: TPS within +/-2% of v775 (no regression) AND
#   nvidia-smi shows reduced GPU memory usage (free headroom for 320K probe).
# If TPS regresses > 5% or quality (3/4 tool-call) drops: rollback.
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
  -v /home/sander/Genesis_Project/vllm_engine/triton-cache-mtp-test:/root/.triton/cache \
  -v /home/sander/Genesis_Project/vllm_engine/compile-cache-prod-mirror-test:/root/.cache/vllm/torch_compile_cache \
  -v /home/sander/genesis-vllm-patches/vllm/_genesis:/usr/local/lib/python3.12/dist-packages/vllm/_genesis:ro \
  -v /home/sander/genesis-vllm-patches/tools/genesis_vllm_plugin:/plugin:ro \
  -v /home/sander/genesis-vllm-patches/external_probe:/external_probe:ro \
  -v "/home/sander/genesis-vllm-patches/vllm/_genesis/configs/moe_tuning/E=256,N=512,device_name=NVIDIA_RTX_A5000,dtype=fp8_w8a8,block_shape=[128,128].json:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/configs/E=256,N=512,device_name=NVIDIA_RTX_A5000,dtype=fp8_w8a8,block_shape=[128,128].json:ro" \
  -e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 -e VLLM_NO_USAGE_STATS=1 \
  -e PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:256,garbage_collection_threshold:0.6" \
  -e VLLM_FLOAT32_MATMUL_PRECISION=high -e NCCL_P2P_DISABLE=1 \
  -e VLLM_USE_FLASHINFER_SAMPLER=1 -e VLLM_USE_FUSED_MOE_GROUPED_TOPK=1 \
  -e OMP_NUM_THREADS=1 -e CUDA_DEVICE_MAX_CONNECTIONS=8 \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 -e TRITON_CACHE_DIR=/root/.triton/cache \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn -e VLLM_MARLIN_USE_ATOMIC_ADD=1 \
  -e VLLM_MOE_USE_DEEP_GEMM=0 -e VLLM_USE_DEEP_GEMM=0 \
  -e VLLM_USE_FLASHINFER_MOE_FP8=0 -e VLLM_LOGGING_LEVEL=WARNING \
  -e GENESIS_ENABLE_P56_SPEC_DECODE_GUARD=0 -e GENESIS_ENABLE_P57_SPEC_DECODE_CAPTURE_SAFE=0 \
  -e GENESIS_ENABLE_PR40768=1 -e GENESIS_ENABLE_PR39055=0 \
  -e GENESIS_ENABLE_PR40738=1 -e GENESIS_ENABLE_PR40738B=1 \
  \
  -e GENESIS_ENABLE_PR36138=1 -e GENESIS_ENABLE_P63_MTP_GDN_STATE_RECOVERY=0 \
  -e GENESIS_ENABLE_PR39598=1 -e GENESIS_ENABLE_P65_TURBOQUANT_SPEC_CG_DOWNGRADE=0 \
  -e GENESIS_ENABLE_P67_TQ_MULTI_QUERY_KERNEL=1 \
  -e GENESIS_P67_USE_UPSTREAM=1 -e GENESIS_P67_NUM_KV_SPLITS=32 \
  -e GENESIS_ENABLE_P68_AUTO_FORCE_TOOL=1 -e GENESIS_ENABLE_P69_LONG_CTX_TOOL_REMINDER=1 \
  -e GENESIS_ENABLE_P70_AUTO_STRICT_NGRAM=1 -e GENESIS_P68_P69_LONG_CTX_THRESHOLD_CHARS=8000 \
  -e GENESIS_ENABLE_P37=1 -e GENESIS_TQ_MAX_MODEL_LEN=320000 \
  -e GENESIS_ENABLE_P72_PROFILE_RUN_CAP=1 -e GENESIS_PROFILE_RUN_CAP_M=4096 \
  -e GENESIS_ENABLE_P74_CHUNK_CLAMP=1 -e GENESIS_ENABLE_PR40610=0 -e GENESIS_ENABLE_PR37629=0 -e GENESIS_ENABLE_PR40925=1 -e GENESIS_ENABLE_P82=1 -e GENESIS_ENABLE_PR40849=1 -e GENESIS_ENABLE_PR40941B=1 -e GENESIS_ENABLE_PR41123=1 -e GENESIS_P82_THRESHOLD_SINGLE=0.3 -e GENESIS_PREALLOC_TOKEN_BUDGET=4096 -e GENESIS_BUFFER_MODE=shared \
  vllm/vllm-openai:nightly -c \
  "set -e; echo \"=== v775 35B baseline upstream P67 (matches v759 PROD) ===\"; \
pip install --quiet --disable-pip-version-check pandas scipy xxhash; \
cp -r /plugin /tmp/genesis_vllm_plugin; \
pip install --quiet --disable-pip-version-check --no-deps -e /tmp/genesis_vllm_plugin 2>&1 | tail -3; \
python3 /external_probe/patch_tolist_cudagraph.py || echo tolist bypass failed; \
python3 /external_probe/patch_40074_iooo.py || echo PR40074 failed; \
python3 -m vllm._genesis.patches.apply_all ; \
exec vllm serve --model /models/Qwen3.6-35B-A3B-FP8 --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.90 --max-model-len 320000 \
  --kv-cache-dtype turboquant_k8v4 --max-num-seqs 2 --max-num-batched-tokens 4096 \
  --enable-chunked-prefill --dtype float16 \
  --disable-custom-all-reduce --language-model-only --trust-remote-code \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
  --api-key genesis-local --served-model-name qwen3.6-35b-a3b --host 0.0.0.0 \
  --speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":3}' \
  --performance-mode interactivity --attention-config.flash_attn_version 2 --port 8000 \
  --no-scheduler-reserve-full-isl --disable-log-stats"
sleep 5
docker logs --tail 5 vllm-server-mtp-test 2>&1 | sed "s/^/  /"
echo "Container started."
