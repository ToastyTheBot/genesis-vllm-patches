# SPDX-License-Identifier: Apache-2.0
"""Genesis Dispatcher v2 — unified patch decision matrix + diagnostics.

Builds on top of `model_detect.py` and `config_detect.py` to provide:

  1. **Per-patch should_apply()** — single-line gate decision for each patch.
     Wraps `config_detect.recommend()` + `model_detect` checks + env-flag
     overrides into one consistent API.

  2. **Apply matrix dump** — diagnostic command-line entry-point that prints
     the full per-patch decision table for the current vllm config. Useful
     for operators to see WHY a patch was applied or skipped without grep-ing
     boot logs.

  3. **Startup logging** — single condensed line at boot summarizing which
     patches got applied, skipped (with reason), or failed. Replaces the
     scattered per-patch INFO lines that flood the boot log.

Usage at runtime
----------------
From a patch wiring (`patch_NN_*.py::apply()`):

    from vllm._genesis.dispatcher import should_apply, log_decision

    decision, reason = should_apply("PR40738")
    if not decision:
        log_decision("PR40738", decision, reason)
        return "skipped", reason
    # ... do the patching ...

From CLI / diagnostic:

    python3 -m vllm._genesis.dispatcher

Output: full apply matrix as ASCII table.

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

log = logging.getLogger("genesis.dispatcher")


# ─── Validation issue dataclass ──────────────────────────────────────────
# Reported by validate_registry() and validate_apply_plan(). Severity
# levels: "ERROR" (operator must fix), "WARNING" (likely-wrong, allow boot
# to proceed), "INFO" (informational only).

@dataclass(frozen=True)
class ValidationIssue:
    severity: str  # "ERROR" | "WARNING" | "INFO"
    patch_id: str
    message: str


# ─── Patch metadata registry ───────────────────────────────────────────────
# Each patch declares what it touches + which env flag enables/disables it.
# This is the SINGLE source of truth for patch-to-feature mapping.

PATCH_REGISTRY: dict[str, dict[str, Any]] = {
    "P56": {
        "title": "TQ spec-decode safe-path guard (deprecated — superseded by P65)",
        "env_flag": "GENESIS_ENABLE_P56_SPEC_DECODE_GUARD",
        "default_on": False,
        "deprecated": True,
        "category": "spec_decode",
        "credit": "noonghunna (#40807, #40831)",
        "upstream_pr": None,
        "conflicts_with": ["P65"],
        "deprecation_note": (
            "P56 was a routing-layer workaround forcing spec-decode through a "
            "'safe' path when CG-aware buffers misaligned. Real fix is P65 "
            "(TurboQuant CG downgrade) which addresses the root cause in the "
            "full-attention path under FULL cudagraph capture. Kept opt-in for "
            "configurations where P65 is intentionally disabled and a routing "
            "guard is still desired (no such config verified in production)."
        ),
    },
    "P57": {
        "title": "TQ spec-decode capture-safe buffers (deprecated — research artifact)",
        "env_flag": "GENESIS_ENABLE_P57_SPEC_DECODE_CAPTURE_SAFE",
        "default_on": False,
        "deprecated": True,
        "category": "spec_decode",
        "credit": "noonghunna (#40831), gdn_attn.py reference",
        "upstream_pr": None,
        "conflicts_with": ["P65"],
        "deprecation_note": (
            "P57 v2 enlarges per-layer capture buffers from ~530 KiB to ~2.1 MiB "
            "(see wiring/patch_57 docstring for derivation), which is the "
            "MINIMAL sufficient fix for the original symptom but pushes total "
            "spec-decode buffer memory from ~270 MiB to ~1080 MiB across 32 "
            "layers — unacceptable on consumer Ampere with 24 GB VRAM. P65 "
            "(CG downgrade) achieves the same correctness without the memory "
            "blow-up. Kept opt-in as a research artifact / reference for "
            "future hardware with larger VRAM budgets."
        ),
    },
    "PR40768": {
        "wiring": "patch_pr40768_async_scheduler_placeholder_fix",
        "title": "Async-scheduler -1 placeholder fix",
        "env_flag": "GENESIS_ENABLE_PR40768_ASYNC_PLACEHOLDER_FIX",
        "env_flag_aliases": ["GENESIS_ENABLE_P58_ASYNC_PLACEHOLDER_FIX"],
        "default_on": False,
        "category": "spec_decode",
        "credit": "z1ying (vllm#40768)",
        "upstream_pr": 40768,
    },
    "PR39055": {
        "wiring": "patch_pr39055_qwen3_reasoning_tool_call_recovery",
        "title": "Qwen3 reasoning embedded tool_call recovery",
        "env_flag": "GENESIS_ENABLE_PR39055_QWEN3_TOOL_RECOVERY",
        "env_flag_aliases": ["GENESIS_ENABLE_P59_QWEN3_TOOL_RECOVERY"],
        "default_on": False,
        "category": "structured_output",
        "credit": "ZenoAFfectionate (vllm#39055)",
        "upstream_pr": 39055,
        "applies_to": {"model_class": ["qwen3", "qwen3_5", "qwen3_moe", "qwen3_next"]},
    },
    "PR40738": {
        "wiring": "patch_pr40738_gdn_ngram_state_recovery",
        "title": "GDN+ngram state recovery (Phase 1: SSM pre-copy)",
        "env_flag": "GENESIS_ENABLE_PR40738_GDN_NGRAM_FIX",
        "env_flag_aliases": ["GENESIS_ENABLE_P60_GDN_NGRAM_FIX"],
        "default_on": False,
        "category": "spec_decode",
        "credit": "tdoublep (vllm#40738), bhaktatejas922 (#39273)",
        "upstream_pr": 40738,
        "applies_to": {"is_hybrid": [True]},
    },
    "PR40738b": {
        "wiring": "patch_pr40738b_gdn_ngram_triton_kernel",
        "title": "GDN+ngram Triton kernel offset (Phase 2)",
        "env_flag": "GENESIS_ENABLE_PR40738B_TRITON_KERNEL",
        "env_flag_aliases": ["GENESIS_ENABLE_P60B_TRITON_KERNEL"],
        "default_on": False,
        "category": "spec_decode",
        "credit": "tdoublep (vllm#40738)",
        "upstream_pr": 40738,
        "applies_to": {"is_hybrid": [True]},
        "requires_patches": ["PR40738"],
    },
    "PR36138": {
        "wiring": "patch_pr36138_structured_output_spec_decode_timing",
        "title": "Structured-output spec-decode reasoning-end timing fix",
        "env_flag": "GENESIS_ENABLE_PR36138_STRUCT_OUT_SPEC_TIMING",
        "env_flag_aliases": ["GENESIS_ENABLE_P62_STRUCT_OUT_SPEC_TIMING"],
        "default_on": False,
        "category": "structured_output",
        "credit": "sfbemerk (vllm#36138), cicirori (vllm#34650)",
        "upstream_pr": 36138,
        "applies_to": {"model_class": ["qwen3", "qwen3_5", "qwen3_6", "qwen3_moe", "qwen3_next"]},
        "conflicts_with": ["PR40962"],
    },
    "P63": {
        "wiring": "patch_63_mtp_gdn_state_recovery",
        "title": "MTP/Eagle drafter GDN state recovery (deprecated — wrong layer)",
        "env_flag": "GENESIS_ENABLE_P63_MTP_GDN_STATE_RECOVERY",
        "default_on": False,
        "deprecated": True,
        "category": "spec_decode",
        "credit": "Genesis-original (hypothesis disproven 2026-04-25)",
        "upstream_pr": None,
        "deprecation_note": (
            "P63 hypothesis was wrong: MTP module uses layer_type='full_attention' "
            "(Qwen3NextAttention), NOT GDN. GDNAttentionMetadataBuilder.build_for_drafting "
            "is never called for MTP drafter. Real fix is P65 (TurboQuant CG downgrade) — "
            "the bug is in the full_attention path under FULL cudagraph capture, not GDN. "
            "P63 may still be relevant for eagle/draft_model methods that use a separate "
            "drafter model with hybrid layers, but no such configuration is verified yet."
        ),
    },
    "P65": {
        "wiring": "patch_65_turboquant_spec_cg_downgrade",
        "title": "TurboQuant spec-decode cudagraph downgrade",
        "env_flag": "GENESIS_ENABLE_P65_TURBOQUANT_SPEC_CG_DOWNGRADE",
        "default_on": False,
        "category": "spec_decode",
        "credit": "Genesis-original (root cause for noonghunna #40880)",
        "upstream_pr": None,
        "applies_to": {"is_turboquant": [True]},
        "conflicts_with": ["P56", "P57", "P67", "P67b"],
    },
    "P68": {
        "wiring": "patch_68_69_long_ctx_tool_adherence",
        "title": "Auto force tool_choice=required for long-context tool calls",
        "env_flag": "GENESIS_ENABLE_P68_AUTO_FORCE_TOOL",
        "default_on": False,
        "category": "structured_output",
        "credit": "Genesis-original (long-ctx tool adherence mitigation)",
        "upstream_pr": None,
        "applies_to": {"model_class": ["qwen3", "qwen3_5", "qwen3_moe", "qwen3_next"]},
    },
    "P69": {
        "title": "Long-context tool-format reminder injection",
        "env_flag": "GENESIS_ENABLE_P69_LONG_CTX_TOOL_REMINDER",
        "default_on": False,
        "category": "structured_output",
        "credit": "Genesis-original (long-ctx tool adherence mitigation)",
        "upstream_pr": None,
        "applies_to": {"model_class": ["qwen3", "qwen3_5", "qwen3_moe", "qwen3_next"]},
    },
    "P70": {
        "wiring": "patch_70_auto_strict_ngram",
        "title": "Auto-strict-ngram (force prompt_lookup_min>=8)",
        "env_flag": "GENESIS_ENABLE_P70_AUTO_STRICT_NGRAM",
        "default_on": False,
        "category": "spec_decode",
        "credit": "Genesis-original (vllm#40875 enforcement)",
        "upstream_pr": None,
    },
    "P67": {
        "wiring": "patch_67_tq_multi_query_kernel",
        "title": "TurboQuant multi-query kernel for spec-decode K+1",
        "env_flag": "GENESIS_ENABLE_P67_TQ_MULTI_QUERY_KERNEL",
        "default_on": False,
        "category": "spec_decode",
        "credit": "Genesis-original (proper fix for noonghunna #40880; replaces P65 workaround). 2026-05-05 NOTE: alternative upstream fix is OPEN as vllm#40914 (Sandermage) — uses synth_seq_lens routing through existing decode kernel instead of new Genesis-original kernel. If #40914 merges, P67 becomes one of two equivalent paths; defer retirement decision until empirical TPS A/B (P67 currently delivers +32% on 35B-A3B-FP8 PROD vs upstream baseline). Watch_for_drift_via vllm#40914.",
        "upstream_pr": None,
        "applies_to": {
            "is_turboquant": [True],
        },
        "conflicts_with": ["P65"],
    },
    "P67b": {
        "wiring": "patch_67b_spec_verify_routing",
        "title": "TurboQuant spec-verify forward() routing (FULL CG enable)",
        # P67b reuses P67's env flag intentionally — they're a coupled pair,
        # P67b is the forward() routing companion that bypasses
        # _prefill_attention for K+1 verify batches (cudagraph-safe).
        "env_flag": "GENESIS_ENABLE_P67_TQ_MULTI_QUERY_KERNEL",
        "default_on": False,
        "category": "spec_decode",
        "credit": "Genesis-original (FULL CG enable for P67 multi-query kernel)",
        "upstream_pr": None,
        "applies_to": {
            "is_turboquant": [True],
        },
        "requires_patches": ["P67"],
        "conflicts_with": ["P65"],
    },
    "P72": {
        "wiring": "patch_72_profile_run_cap",
        "title": "profile_run M cap (unblocks --max-num-batched-tokens>4096 on MoE)",
        "env_flag": "GENESIS_ENABLE_P72_PROFILE_RUN_CAP",
        "default_on": False,
        "category": "compile_safety",
        "credit": "Genesis-original (Dynamo fake-tensor mismatch workaround for moe_align_block_size symbolic shape)",
        "upstream_pr": None,
    },
    "PR40819": {
        "wiring": "patch_pr40819_block_verify",
        "title": "Block-verify rejection sampler (Sun 2024 ICLR)",
        "env_flag": "GENESIS_ENABLE_PR40819_BLOCK_VERIFY",
        "env_flag_aliases": ["GENESIS_ENABLE_P71_BLOCK_VERIFY"],
        "default_on": False,
        "category": "spec_decode",
        "credit": "Backport of vllm#40819 (Z. Golpayegani draft) + Sun et al. arXiv 2403.10444 + 2 critical fixes from gemini-code-assist review (shared u per request, denom==0 → 1.0)",
        "upstream_pr": 40819,
    },
    "P74": {
        "wiring": "patch_74_chunk_clamp",
        "title": "Auto chunk-clamp via long_prefill_token_threshold (P72 companion)",
        "env_flag": "GENESIS_ENABLE_P74_CHUNK_CLAMP",
        "default_on": False,
        "category": "compile_safety",
        "credit": "Genesis-original (zero-VRAM-cost prealloc-overflow safety net for P72-unblocked batched_tokens>4096)",
        "upstream_pr": None,
        "applies_to": {"model_class": ["qwen3", "qwen3_5", "qwen3_moe", "qwen3_next"]},
        "requires_patches": ["P72"],
    },
    "PR25784": {
        "wiring": "patch_pr25784_suffix_decoding_enable",
        "title": "Auto-enable Suffix Decoding (Arctic Inference, vllm#25784)",
        "env_flag": "GENESIS_ENABLE_PR25784_SUFFIX_DECODING",
        "env_flag_aliases": ["GENESIS_ENABLE_P75_SUFFIX_DECODING"],
        "default_on": False,
        "category": "spec_decode",
        "credit": "Backport-enabler of vllm#25784 (Arctic Inference Suffix Decoding) — operator convenience: auto-swap method=ngram→suffix when env enabled. Algorithm: arxiv 2411.04975.",
        "upstream_pr": 25784,
    },
    "P77": {
        "wiring": "patch_77_adaptive_ngram_k",
        "title": "Adaptive ngram K controller (EMA + hysteresis + auto-disable)",
        "env_flag": "GENESIS_ENABLE_P77_ADAPTIVE_NGRAM_K",
        "default_on": False,
        "category": "spec_decode",
        "credit": "Genesis-original (port of SGLang adaptive_spec_params.py EMA+hysteresis Apache-2.0 + Nightjar arXiv 2512.22420 auto-disable extension). Targets free-form ngram pathology (46 tok/s).",
        "upstream_pr": None,
    },
    "P78": {
        "wiring": "patch_78_tolist_capture_guard",
        "title": "TurboQuant .tolist() capture-guard (adapted from noonghunna)",
        "env_flag": "GENESIS_ENABLE_P78_TOLIST_CAPTURE_GUARD",
        "default_on": False,
        "category": "compile_safety",
        "credit": "Adapted from noonghunna's patch_tolist_cudagraph.py (Apache-2.0, github.com/noonghunna/qwen36-27b-single-3090). Surgical safety-net for cudagraph capture; complements our P22/P26/P44 prealloc.",
        "upstream_pr": None,
        "applies_to": {
            "is_turboquant": [True],
            "quant_format": ["fp8", "compressed_tensors"],
        },
    },
    "PR40610": {
        "wiring": "patch_pr40610_async_proposer_sync",
        "title": "Async × spec-decode proposer-sync backport (vllm#40610)",
        "env_flag": "GENESIS_ENABLE_PR40610_ASYNC_PROPOSER_SYNC",
        "env_flag_aliases": ["GENESIS_ENABLE_P79B_ASYNC_PROPOSER_SYNC"],
        "default_on": False,
        "category": "spec_decode",
        "credit": "Backport of vllm#40610 (OPEN draft, tracked from #40608). Re-records prepare_inputs_event AFTER spec-decode proposer GPU work in sample_tokens(). Fixes async × spec-decode race where next batch _update_states could mutate block_table while previous batch's proposer was still reading on GPU. Genesis prod uses sync ngram so direct value is minimal; protects users on async+EAGLE/MTP/ngram_gpu.",
        "upstream_pr": 40610,
    },
    "PR37629": {
        "wiring": "patch_pr37629_stale_spec_token_cleanup",
        "title": "Stale spec_token_ids cleanup for unscheduled requests (vllm#37629)",
        "env_flag": "GENESIS_ENABLE_PR37629_STALE_SPEC_TOKEN_CLEANUP",
        "env_flag_aliases": ["GENESIS_ENABLE_P79C_STALE_SPEC_TOKEN_CLEANUP"],
        "default_on": False,
        "category": "spec_decode",
        "credit": "Backport of vllm#37629 (OPEN, fixes #36906). Cleanup pass after main scheduling loop clears spec_token_ids for unscheduled running requests. Prevents -1 placeholder leak into F.embedding() under budget-exhausted high-concurrency on async + EAGLE/MTP. Genesis prod (max_num_seqs=2, sync ngram) gains nothing direct; protects high-concurrency multimodal users.",
        "upstream_pr": 37629,
    },
    "PR40925": {
        "wiring": "patch_pr40925_fp8_block_scaled_m_le_8",
        "title": "fp8 block-scaled MM low-M decode tuning (vllm#40925)",
        "env_flag": "GENESIS_ENABLE_PR40925_FP8_BLOCK_SCALED_M_LE_8",
        "env_flag_aliases": ["GENESIS_ENABLE_P81_FP8_BLOCK_SCALED_M_LE_8"],
        "default_on": False,
        "category": "kernel_perf",
        "credit": "Backport of vllm#40925 (tonyliu312, OPEN). Specializes w8a8_triton_block_scaled_mm default config for M<=8 (single-request decode + MTP K=3 verify): BLOCK_SIZE_M 64->16, num_stages 2->3 (non-ROCm). Empirical +23% median decode on GB10. Direct hit for Genesis prod (Qwen3.6-A3B FP8 + max_num_seqs=2 + no pre-tuned JSON for A5000).",
        "upstream_pr": 40925,
        "applies_to": {
            "quant_format": ["fp8"],
        },
    },
    "P82": {
        "wiring": "patch_82_sglang_acceptance_threshold",
        "title": "SGLang threshold_single OR-clause acceptance (BIASED — opt-in research)",
        "env_flag": "GENESIS_ENABLE_P82",
        "default_on": False,
        "category": "spec_decode",
        "credit": "SGLang team (sgl-project/sglang) speculative_sampling.cuh — port of the threshold_single OR-clause that breaks the structural ceiling clean_rate ≈ accept_rate^num_spec. Targets v7.13 strict-ngram acceptance gap. BIASED rule (loses unbiased-sampling guarantee); requires empirical quality validation before prod. Threshold baked from env GENESIS_P82_THRESHOLD_SINGLE (default 0.3) at server start.",
        "upstream_pr": None,
    },
    "P83": {
        "wiring": "patch_83_mtp_keep_last_cached_block",
        "title": "MTP keep-last-cached-block (vllm#38182 downstream symptom — P84 is real fix)",
        "env_flag": "GENESIS_ENABLE_P83",
        "default_on": False,
        "category": "spec_decode",
        "credit": "Root-cause analysis: vllm#38182 by uOnePiece + @Angazenn comment identifying single_type_kv_cache_manager.py:457 force-pop last cached block when use_eagle=True. MTP gets caught up via config/speculative.py:890-891 (use_eagle returns True for 'mtp'). EMPIRICALLY DISPROVEN as the actual cause: Genesis debug instrumentation showed find_longest_cache_hit was NEVER called for our workload because num_hashes=0 (block_size > prompt_len after P5 LCM-pad). The L457 pop is a downstream symptom, not the upstream cause. P84 (hash_block_size override) is the real fix. P83 kept as opt-in research artifact for future workloads where the pop site IS reached.",
        "upstream_pr": None,
        "applies_to": {"is_hybrid": [True]},
    },
    "P84": {
        "wiring": "patch_84_hash_block_size_override",
        "title": "hash_block_size override (vllm#38182 actual root cause)",
        "env_flag": "GENESIS_ENABLE_P84",
        "default_on": False,
        "category": "kv_cache",
        "credit": "Genesis-original discovery 2026-04-27 via P83 DEBUG instrumentation. scheduler.py:234 hard-codes hash_block_size=self.block_size; on hybrid Qwen3.6-MoE with P5 LCM-pad this becomes 2048+, so request_block_hasher computes 0 hashes for prompts < 2048 tokens. Cache machinery runs with overhead but never produces hits. P84 text-patches scheduler.py to read hash_block_size from env GENESIS_P84_HASH_BLOCK_SIZE (recommended value: 16 = full-attention default). Engage via GENESIS_ENABLE_P84=1 + GENESIS_P84_HASH_BLOCK_SIZE=16. Constraint: must divide every group's block_size, else vLLM's own assertion fires at startup. Related: vllm#38182 identified WRONG root cause (the L457 pop); P84 attacks the upstream cause.",
        "upstream_pr": None,
        "applies_to": {"is_hybrid": [True]},
    },
    "P85": {
        "wiring": "patch_85_hybrid_fine_shadow_prefix_cache",
        "title": "Hybrid fine-shadow prefix cache (vllm#38182 followup, MambaManager fix)",
        "env_flag": "GENESIS_ENABLE_P85",
        "default_on": False,
        "category": "kv_cache",
        "credit": "Genesis-original 2026-04-27 — synthesis of 6-round empirical investigation + deep code analysis. Identified TWO mismatches in hybrid prefix cache: (A) MambaManager.cache_blocks early-returns for prompts < self.block_size (e.g., 1424 < 2048); (B) Mamba align-mode pads with null_blocks so num_full_blocks > 0 still inserts 0 entries. P85 patches MambaManager to: (1) register shadow fine-grained hash entries (scale_factor=block_size/hash_block_size duplicates) when caching, (2) walk fine hashes on lookup with eviction-safety re-derive verify. Memory layout / ref-count untouched. Requires P84 (fine hashes computed). Architectural limit: cannot help prompts < block_size (Mamba state genuinely uncached at sub-block boundaries).",
        "upstream_pr": None,
        "applies_to": {"is_hybrid": [True]},
        "requires_patches": ["P84"],
    },
    "PR40849": {
        "wiring": "patch_pr40849_mtp_draft_online_quant_propagation",
        "title": "MTP/draft online-quant propagation (vllm#40849)",
        "env_flag": "GENESIS_ENABLE_PR40849_MTP_DRAFT_ONLINE_QUANT",
        "env_flag_aliases": ["GENESIS_ENABLE_PN8_MTP_DRAFT_ONLINE_QUANT"],
        "default_on": False,
        "category": "spec_decode",
        "credit": (
            "Backport of vllm#40849 (bhoomit, OPEN). Modifies "
            "`get_draft_quant_config()` so that, when the spec-decode draft "
            "model has no explicit quantization config, it inherits the "
            "target's `OnlineQuantizationConfig` (e.g. fp8_per_tensor). "
            "Frees ~600 MiB on FP8-target + Eagle3 / DFlash / MTP-as-external-"
            "draft worker (1.45 GiB BF16 → 0.88 GiB FP8 on Qwen3-32B + Eagle3 "
            "per PR author bench). Also catches ValueError/FileNotFoundError "
            "in the existing draft lookup path (online-quant methods crash "
            "through checkpoint-config because hf_overrides is callable). "
            "NO-OP for current Genesis prod (Lorbus/Minachist 27B do not run "
            "online-quant + external draft). Becomes valuable when DFlash / "
            "Eagle3 / FP8 stacks roll out."
        ),
        "upstream_pr": 40849,
        "applies_to": {
            # Predicate enforced naturally by the patched function — when
            # spec-decode is off OR target is not online-quantized, the new
            # branch falls through identical to vanilla. No model gating.
        },
    },
    "PR39930": {
        "wiring": "patch_pr39930_independent_drafter_attn_backend",
        "title": "Independent drafter attention backend (vllm#39930)",
        "env_flag": "GENESIS_ENABLE_PR39930_INDEPENDENT_DRAFTER_ATTN",
        "env_flag_aliases": ["GENESIS_ENABLE_PN9_INDEPENDENT_DRAFTER_ATTN"],
        "default_on": False,
        "category": "spec_decode",
        "credit": (
            "Backport of vllm#39930 (MatthewBonanni, MERGED). Allows the "
            "spec-decode drafter to use a different attention backend than "
            "the target model. Unblocks drafters with incompatible "
            "requirements (e.g. DFlash needs non-causal attention support, "
            "which TRITON_ATTN does not provide → ValueError on boot). "
            "Modifies `LLMBaseProposer._create_draft_vllm_config()` to "
            "always reset the drafter's attention backend (None = "
            "auto-select). Genesis minimal port: env "
            "GENESIS_PR39930_DRAFTER_BACKEND chooses backend (e.g. FLASH_ATTN); "
            "unset/auto → drafter auto-selects. Does NOT add the new "
            "SpeculativeConfig.attention_backend pydantic field (too "
            "invasive at runtime for a frozen dataclass + field_validator). "
            "Unblocks DFlash spike sprint task without full pin bump risk "
            "from #40860 mega-merge. NO-OP for current Genesis prod (PROD "
            "uses ngram drafter, no attention backend conflict)."
        ),
        "upstream_pr": 39930,
        "applies_to": {
            # Patch only takes effect inside _create_draft_vllm_config which
            # is only called when spec-decode is active. No additional gate.
        },
    },
    "PR40425": {
        "wiring": "patch_pr40425_dflash_quant_drafter",
        "title": "DFlash drafter quantization support (PR #40425 backport)",
        "env_flag": "GENESIS_ENABLE_PR40425_DFLASH_QUANT_DRAFTER",
        "env_flag_aliases": ["GENESIS_ENABLE_PN38_DFLASH_QUANT_DRAFTER"],
        "default_on": False,
        "category": "spec_decode",
        "credit": (
            "Backport of vllm-project/vllm#40425 (infatoshi, OPEN). "
            "Enables quantized DFlash drafter checkpoints (FP8 W8A8, "
            "NVFP4, AWQ, etc.) — correctness/compat fix per PR title, "
            "NOT throughput improvement claim. Today NO-OP for our BF16 "
            "drafters in /nfs/genesis/models/Qwen3.6-{27B,35B-A3B}-DFlash "
            "(quant_config is None → original dense fast-path runs). "
            "Tomorrow: drop-in support for FP8/NVFP4 drafter checkpoints "
            "(e.g. AEON-7/Qwen3.6-NVFP4-DFlash, llm-compressor self-quant). "
            "Memory savings on adoption: BF16 drafter ~2.4 GB → FP8 ~1.2 GB "
            "per worker, ~2.4 GB total at TP=2 — frees KV-cache headroom. "
            "4 sub-patches into qwen3_dflash.py (Site A: F.linear→qkv_proj; "
            "B: pass quant_config to layer; C: conditional fused-KV; D: "
            "quantized fallback in precompute). Composable с PN40-A "
            "(different anchor surfaces in same file)."
        ),
        "upstream_pr": 40425,
        "applies_to": {
            "spec_method": ["dflash"],
        },
    },
    "PN40-classifier": {
        "title": "PN40 sub-D workload classifier (chat_completion middleware)",
        "env_flag": "GENESIS_ENABLE_PN40_DFLASH_OMNIBUS",
        "default_on": False,
        "category": "spec_decode",
        "credit": (
            "Genesis-original 2026-05-04 — companion to PN40 sub-D. "
            "Text-patches vllm/entrypoints/openai/chat_completion/serving.py "
            "(audit A-13 fix 2026-05-05: was incorrectly listed as "
            "serving_chat.py — actual target is chat_completion/serving.py) "
            "to classify each request as code/short_ctx/long_ctx/free_form "
            "and stash on `request._genesis_pn40_workload_class`. "
            "Consumer is the runtime K-trim hook in PN40 sub-C "
            "(scheduler.update_draft_token_ids). Toggled jointly with "
            "PN40 master via GENESIS_ENABLE_PN40_DFLASH_OMNIBUS — no "
            "separate enable flag (sub-D is universal companion to sub-C). "
            "Tier bias: code +1, long_ctx -1, others 0. Defensive on "
            "unknown class names (falls through to neutral bias)."
        ),
        "upstream_pr": None,
        "applies_to": {
            "spec_method": ["dflash", "mtp"],
        },
    },
    "PN40": {
        "wiring": "patch_N40_dflash_omnibus",
        "title": "Spec-decode omnibus (A DFlash K-norm + B pool + C adaptive K + D sentinel)",
        "env_flag": "GENESIS_ENABLE_PN40_DFLASH_OMNIBUS",
        "default_on": False,
        "category": "spec_decode",
        "credit": (
            "Genesis-original 2026-05-04 — 4-component omnibus spec-decode "
            "optimization with strict no-regression contract. "
            "Sub-A (DFlash-only): fused per-layer K-norm Triton kernel "
            "replaces L-iteration loop in qwen3_dflash.py:397-404. "
            "Numerical TDD 12/12 PASS rel_avg=0.0000. Microbench vs "
            "_custom_ops.rms_norm: 3.22x (27B L=5) / 5.32x (35B L=8). "
            "Per-draft-step saving +37us (27B) / +70us (35B). "
            "Sub-B (DFlash-only MVP): persistent K/V buffer pool, "
            "LRU-bounded, hit-rate tracked. Saves cudaMalloc churn. "
            "Sub-C (UNIVERSAL): adaptive K/N controller, mirrors SGLang "
            "tier policy + EMA hysteresis. Default tiers MTP K=3 [0,1,3], "
            "DFlash N=5 [0,1,3,5], DFlash N=3 [0,1,3]. NaN-trip safety. "
            "Applies to ALL 4 configs (27B/35B x MTP/DFlash). "
            "Sub-D (UNIVERSAL): workload classifier (code/short/long/"
            "free-form) + stability sentinel (sliding-window AL drop "
            "detector + NaN trip). Applies to ALL 4 configs. "
            "TDD: 12/12 (sub-A numerical) + 35/35 (sub-B/C/D logic). "
            "Per-sub env toggles GENESIS_PN40_ENABLE_SUB_{A,B,C,D}=0 "
            "to disable individually. Strict-superset throughout: "
            "any eligibility failure falls through to baseline. Default "
            "OFF master-gated until A/B prod-validates. Composes "
            "additively with PR40898/PN23/PR40727/P77."
        ),
        "upstream_pr": None,
        "applies_to": {
            "spec_method": ["dflash", "mtp"],  # C+D universal across both
        },
    },
    # PN37 archived 2026-05-04 to vllm/_genesis/_not_used_artifact/.
    # Premise (FA2 dead-zone for tiny-Q non-causal) was disproved by
    # microbench: torch SDPA already routes to FA2 packed-GQA path well.
    # Kernel + TDD (rel_avg < 0.01) preserved as research artifact;
    # entry intentionally NOT in PATCH_REGISTRY (no dispatcher matrix
    # row, no apply_all skip-noise on every boot).
    # PN36 was removed 2026-05-04 — was a misdiagnosis. The 5 `self.reasoner`
    # call-sites I found were inserted by OUR PR36138 backport (vllm#36138, still
    # OPEN), NOT by upstream. Upstream PR #41199 (MERGED 2026-05-01, included
    # in our pin) intentionally moved reasoner to per-request lazy build via
    # `self._get_reasoner(request)`. Pristine upstream code does NOT reference
    # `self.reasoner`. Fix path: disable PR36138 on this pin (it collides with
    # the rename refactor) and backport PR #40962 separately for the
    # post-reasoning-boundary spec-decode case.
    "PN50": {
        "wiring": "patch_N50_gdn_fused_proj",
        "title": "GDN proj fusion (SGLang#21019 backport — Qwen3.5/3.6 contiguous-projection Triton kernel)",
        "env_flag": "GENESIS_ENABLE_PN50_GDN_FUSED_PROJ",
        "default_on": False,
        "category": "perf_kernel",
        "credit": (
            "Backport of SGLang PR #21019 (MERGED 2026-03-23, commit "
            "5bdc07d). Original Triton kernel by Yuan Luo (@yuan-luo), "
            "Apache-2.0. Replaces the unfused split/reshape/cat/.contiguous() "
            "chain (5-6 launches + 2 explicit copies) in `gdn_linear_attn.py:562-566` "
            "Qwen3.5/3.6 contiguous projection branch with single fused Triton "
            "kernel `fused_qkvzba_split_reshape_cat_contiguous` (310 LOC). "
            "Pure data-copy kernel — no math, no reductions, no numerical drift. "
            "Output layout bit-identical to unfused PyTorch. Wrapper falls "
            "through to PyTorch reference on: non-contiguous input, non-pow2 "
            "head_dim, V_PER_GROUP non-integer, kernel launch failure. "
            "Affects: 27B Lorbus only (35B is Qwen3MoE — no GDN layers, "
            "patch never fires). Claimed gain on H200/Qwen3.5-35B-A3B (SGLang "
            "naming): +7.4% TPS, -10.8% TTFT, -31.2% ITL PR40385. On A5000 + "
            "27B Lorbus expect modest gain (memory-bound layer, A5000 PCIe "
            "slower than H200). Composable with PR41142/PR41446/PN32/P103 — verified "
            "no overlap (PR41142 acts in interleaved branch, others in different "
            "files). Default OFF until live A/B prod-validates."
        ),
        "upstream_pr": None,
        "applies_to": {
            "model_class": ["qwen3_5", "qwen3_6"],
        },
    },
    "PN59": {
        "wiring": "patch_N59_streaming_gdn",
        "title": "Streaming-GDN orchestrator (Variant D Phase 2) — true Cliff 2b OOM fix",
        "env_flag": "GENESIS_ENABLE_PN59_STREAMING_GDN",
        "default_on": False,
        "category": "hybrid",
        "credit": (
            "Genesis-original 2026-05-05, Variant D Phase 2. Replaces the "
            "(B, NT, H, V, K) full materialization in `chunk_gated_delta_rule_"
            "fwd_h → chunk_fwd_o` consumer pair with window-iterative driver. "
            "Eliminates Cliff 2b multi-turn OOM (Issue #19) — root cause: 805 "
            "MiB single allocation per layer per forward at T=64K Genesis 27B "
            "Lorbus shapes. Cross-engine validation: llama.cpp + MLX-LM use "
            "pure-streaming register-resident state, survive multi-turn; "
            "vLLM/SGLang/FLA materialize-full, hit Cliff 2b. **Independent "
            "confirmation** by noonghunna (issue #20, 2026-05-05): 'the "
            "limitation is the triton kernel for cliff 2; doesn't appear with "
            "llama.cpp'. Phase 1 numerical TDD proves bit-equivalence "
            "(rtol<1e-5) on 10 Genesis 27B shape cases. Composes with "
            "PN50/PN54/PR41422/P67 (orthogonal). Supersedes P103 outer chunked "
            "wrapper when both ON. Default OFF until live A/B prod-validates."
        ),
        "upstream_pr": None,  # FLA RFC #485, #190 pending — first-mover position
        "applies_to": {
            "model_class": ["qwen3_5", "qwen3_6"],  # 27B Lorbus hybrid only
        },
        "conflicts_with": [],
    },
    "PR40962": {
        "wiring": "patch_pr40962_spec_reasoning_boundary",
        "title": "Spec-decode reasoning boundary validation — narrower alt to PR36138 (vllm#40962)",
        "env_flag": "GENESIS_ENABLE_PR40962_SPEC_REASONING_BOUNDARY",
        "env_flag_aliases": ["GENESIS_ENABLE_PN58_SPEC_REASONING_BOUNDARY"],
        "default_on": False,
        "category": "structured_output",
        "credit": (
            "Backport of vllm#40962 (OPEN, AI-assisted by author). NARROWER "
            "alternative to our existing PR36138 (vllm#36138 broader pipeline-"
            "level fix). MUTUALLY EXCLUSIVE with PR36138 — both patch the same "
            "`should_advance` block in scheduler.update_from_output(). "
            "Apply check enforces PR36138 OFF requirement; SKIPS otherwise. "
            "PR40962 modifies ONLY commit-time validation, doesn't touch "
            "bitmask/draft validation. Author warns: significant perf drop "
            "with multi-token reasoning markers (per-token boundary scan "
            "expensive). Engineering tradeoff: PR36138 = more correct (per-pos "
            "grammar masks), more invasive; PR40962 = less correct in edge "
            "cases (commit-time only), cheaper hot-path. Multi-file "
            "(envs.py + abs_reasoning_parsers.py + basic_parsers.py + "
            "v1/structured_output/__init__.py + v1/core/sched/scheduler.py "
            "= 5 files, 6 sub-patches). Default OFF; current Genesis PROD "
            "uses PR36138 (broader). Enable PR40962 only after measuring PR36138 perf "
            "hit on YOUR specific reasoning parser."
        ),
        "upstream_pr": 40962,
        "applies_to": {},
        "conflicts_with": ["PR36138"],
    },
    "PR41467": {
        "wiring": "patch_pr41467_mtp_truncation_detector",
        "title": "MTP truncation detector at reasoning→tool_call boundary (vllm#41467)",
        "env_flag": "GENESIS_ENABLE_PR41467_MTP_TRUNCATION_DETECTOR",
        "env_flag_aliases": ["GENESIS_ENABLE_P107_MTP_TRUNCATION_DETECTOR"],
        "default_on": False,
        "category": "structured_output",
        "credit": (
            "Backport of vllm#41467 (ToastyTheBot, OPEN). При MTP K≥1 + "
            "tools + reasoning_parser возможна редкая (~0.25% per author "
            "on Qwen3.6 27B-FP8) ситуация: модель производит EOS на "
            "boundary reasoning→tool_call. finish_reason=stop, ни "
            "tool_calls, ни content. Defensive guard в "
            "chat_completion_stream_generator detect'ит combo и raise "
            "GenerationError (retryable) → клиент retries вместо silent "
            "stop. Author явно ссылается на наш PR40768/PR39055/PR40738/P61/PR39598 path. "
            "EXACT наш PROD config (27B Lorbus + MTP K=3 + tools). Defensive "
            "safety-net, не root-cause fix. Default OFF до live verify "
            "tool-call sweep на 27B PROD."
        ),
        "upstream_pr": 41467,
        "applies_to": {},
    },
    "PR41418b": {
        "wiring": "patch_pr41418b_tq_centroids_disk_cache",
        "title": "TurboQuant centroids disk-persistent cache (vllm#41418-inspired)",
        "env_flag": "GENESIS_ENABLE_PR41418B_TQ_CENTROIDS_DISK_CACHE",
        "env_flag_aliases": ["GENESIS_ENABLE_PN57_TQ_CENTROIDS_DISK_CACHE"],
        "default_on": False,
        "category": "perf_hotfix",
        "credit": (
            "Inspired by vllm#41418 (TheTom, OPEN). Upstream PR pre-bakes 9 "
            "(d,bits) centroid tables inline (~1500 LOC of constants). "
            "Genesis approach: disk-persistent cache `~/.cache/genesis/"
            "turboquant_centroids.pkl` instead of inline constants. "
            "Lloyd-Max solver fully deterministic given (d,bits) → bit-"
            "identical to upstream pre-baked tables. Cold start: 200ms × N "
            "first-time shapes per fresh container. Subsequent boots / "
            "worker restarts: instant lookup. Saves ~205ms per worker on "
            "k8v4 path. Atomic write (tempfile+rename), defensive fall-"
            "through to solver on any cache failure. Default OFF until "
            "live-verified cold-start savings."
        ),
        "upstream_pr": 41418,
        "applies_to": {},
    },
    "PR41602": {
        "wiring": "patch_pr41602_wake_up_hybrid_kv",
        "title": "wake_up crash fix on hybrid (Mamba/DeltaNet) models — vllm#41602 backport",
        "env_flag": "GENESIS_ENABLE_PR41602_WAKE_UP_HYBRID_KV",
        "env_flag_aliases": ["GENESIS_ENABLE_PN55_WAKE_UP_HYBRID_KV"],
        "default_on": False,
        "category": "perf_hotfix",
        "credit": (
            "Backport of vllm-project/vllm#41602 (OPEN as of 2026-05-04, "
            "AI-assisted by author kevglynn). `init_fp8_kv_scales()` "
            "AttributeError на Mamba/DeltaNet hybrid после `/sleep` → "
            "`/wake_up`. MambaSpec stores per-layer state as `list[Tensor]` "
            "not single tensor; original loop naively called `.zero_()` on "
            "list → AttributeError ломает entire wake-up. Fix: isinstance "
            "check + iterate over list. Affects 27B Lorbus Qwen3.6 hybrid "
            "(GDN = MambaSpec layers). Crash trigger: any /sleep+/wake_up "
            "via management API. Genesis active scripts don't use sleep, "
            "but defensive backport recommended for any external mgmt-API "
            "trigger. Default OFF; enable when sleep/wake actively used."
        ),
        "upstream_pr": 41602,
        "applies_to": {},
    },
    "PN54": {
        "wiring": "patch_N54_gdn_contiguous_dedup",
        "title": "GDN contiguous-call deduplication (P0.7 Cliff 2b OOM mitigation)",
        "env_flag": "GENESIS_ENABLE_PN54_GDN_CONTIGUOUS_DEDUP",
        "default_on": False,
        "category": "perf_hotfix",
        "credit": (
            "Genesis-original 2026-05-04, inspired by MLX-LM PR #1077 "
            "(adurham, MIT) root-cause analysis: shared-buffer/slice-keeps-"
            "parent-alive class of bug. Removes 2 redundant `.contiguous()` "
            "calls in `gdn_linear_attn.py` already guaranteed contiguous by "
            "operator semantics OR re-enforced by FLA `@input_guard`. "
            "Sub-A: `ssm_state[non_spec_state_indices_tensor].contiguous()` "
            "(line ~985) — advanced index already produces fresh allocation; "
            "saves one full ssm_state-shape copy per prefill batch. Sub-B: "
            "LoRA branch `b/a.contiguous()` after `chunk(2, -1)` (lines 551-"
            "552) — chunk on last dim returns contiguous halves; LoRA-only, "
            "no-op on Genesis non-LoRA PROD. Target: Cliff 2b multi-turn "
            "OOM (Genesis Issue #19) — observed +1400 MiB/turn allocator "
            "delta; estimated saving 300-600 MiB/turn. Models: 27B Lorbus "
            "INT4 (all configs with GDN) — sub-A fires; 35B Qwen3MoE no GDN "
            "— patch never fires. Default OFF until live A/B Cliff 2b "
            "reproducer shows per-turn delta drops below ~900 MiB."
        ),
        "upstream_pr": None,
        "applies_to": {
            "model_class": ["qwen3_5", "qwen3_6"],
        },
    },
    "PR41411": {
        "wiring": "patch_pr41411_prompt_logprobs_eviction",
        "title": "prompt_logprobs eviction fix during chunked prefill (vllm#41411 backport)",
        "env_flag": "GENESIS_ENABLE_PR41411_PROMPT_LOGPROBS_EVICTION",
        "env_flag_aliases": ["GENESIS_ENABLE_PN52_PROMPT_LOGPROBS_EVICTION"],
        "default_on": False,
        "category": "perf_hotfix",
        "credit": (
            "Backport of vllm-project/vllm#41411 (MERGED 2026-05-04 18:46 UTC "
            "by Joachim Studnia, Mistral). Fixes TWO bugs in v1 gpu_worker "
            "prompt_logprobs path: (1) overly aggressive `-1` in "
            "`includes_prompt = computed_prefill < prompt_lens - 1` skipped "
            "the last prompt token's logprob when chunked-prefill boundary "
            "fell on `prompt_lens - 1`; (2) `in_progress_prompt_logprobs_cpu` "
            "stored on `input_batch` per-batch dict was lost on request "
            "eviction → silent corruption / IndexError on re-schedule. "
            "Multi-file text-patch: prompt_logprob.py + gpu_input_batch.py "
            "(field move) + gpu_model_runner.py (read/write per-request). "
            "Affects Genesis configs that combine `--enable-chunked-prefill` "
            "(all of ours) + spec-decode (MTP K=3 on PROD) + clients passing "
            "`prompt_logprobs=N`. Default OFF until live verify with Open "
            "WebUI / LibreChat workload that exercises prompt_logprobs."
        ),
        "upstream_pr": 41411,
        "applies_to": {},
    },
    "PN51": {
        "wiring": "patch_N51_qwen3_streaming_thinking_disabled",
        "title": "Qwen3 streaming `enable_thinking=false` content routing (vllm#40816 backport)",
        "env_flag": "GENESIS_ENABLE_PN51_QWEN3_STREAMING_THINKING_DISABLED",
        "default_on": False,
        "category": "perf_hotfix",
        "credit": (
            "Backport of upstream issue vllm-project/vllm#40816 (OPEN, filed "
            "2026-04-22 by 'keehawkes'). When server is launched with "
            "--default-chat-template-kwargs '{\"enable_thinking\": false}' or "
            "the request passes chat_template_kwargs.enable_thinking=false, "
            "streaming responses incorrectly route every model token via "
            "delta.reasoning instead of delta.content. Mirrors the existing "
            "non-streaming short-circuit at qwen3_reasoning_parser.py:146-148. "
            "Affects ALL OpenAI-compatible streaming clients that read "
            "delta.content (Open WebUI, LibreChat, LobeChat, Cline, OpenCode). "
            "Single-line guard at extract_reasoning_streaming entry; no risk "
            "for thinking-enabled requests (guard False). Default OFF until "
            "Open WebUI / LibreChat repro proves the fix end-to-end on "
            "Genesis 27B/35B + reasoning-parser qwen3."
        ),
        "upstream_pr": 40816,
        "applies_to": {},
    },
    "PR35975": {
        "wiring": "patch_pr35975_inputs_embeds_optional",
        "title": "Skip inputs_embeds buffer for text-only models (vllm#35975 backport)",
        "env_flag": "GENESIS_ENABLE_PR35975_INPUTS_EMBEDS_OPTIONAL",
        "env_flag_aliases": ["GENESIS_ENABLE_PN35_INPUTS_EMBEDS_OPTIONAL"],
        "default_on": True,
        "category": "perf_hotfix",
        "credit": (
            "Backport of vllm-project/vllm#35975 by AjAnubolu (OPEN since "
            "2026-03-04). Skips the (max_num_tokens, hidden_size) GPU "
            "buffer allocation for text-only models in BOTH "
            "gpu_model_runner (~64 MiB GPU + 64 MiB pinned CPU) AND "
            "llm_base_proposer spec-decode proposer (~64 MiB GPU). For "
            "Qwen3.6-27B at max_num_tokens=4096 and hidden_size=8192: "
            "freed ~128 MiB GPU + 64 MiB pinned CPU per worker. "
            "Particularly relevant on borderline-OOM single-24GB-GPU "
            "configs (Cliff 2 fires at 50 MiB-free thresholds) and "
            "WSL2 setups with extra display overhead. Pattern credit: "
            "noonghunna club-3090 setup-time sidecar "
            "patch_inputs_embeds_optional.py 2026-05-02. Originally "
            "raised by club-3090#32 (RossNE99, GuiPerPT WSL2 OOM "
            "reports). Default ON — strict memory savings, no "
            "regression possible (the `if` guard preserves original "
            "allocation for multimodal models). Auto-retires when "
            "vllm#35975 merges upstream."
        ),
        "upstream_pr": 35975,
    },
    "PR37521": {
        "wiring": "patch_pr37521_spec_decode_warmup_k",
        "title": "Spec-decode warmup K-aware sizing (vllm#37521 extended to MTP/ngram)",
        "env_flag": "GENESIS_ENABLE_PR37521_SPEC_DECODE_WARMUP_K",
        "env_flag_aliases": ["GENESIS_ENABLE_PN33_SPEC_DECODE_WARMUP_K"],
        "default_on": True,
        "category": "spec_decode",
        "credit": (
            "Backport of vllm-project/vllm#37521 (itailang, OPEN at "
            "backport time 2026-05-02) EXTENDED beyond use_eagle() "
            "gate to cover all spec-decode methods (EAGLE + MTP + "
            "ngram + draft-model). Root-cause fix: gpu_model_runner."
            "_dummy_sampler_run() warmed up with dummy K=1 instead of "
            "real num_speculative_tokens, causing (a) KV-cache profile "
            "to over-estimate available headroom → mid-stream OOM via "
            "propose_draft_token_ids (ampersandru, club-3090#16 "
            "2026-05-01) AND (b) TurboQuant WorkspaceManager lock fails "
            "when real spec-decode tries to grow workspace beyond "
            "warmup-reserved size (noonghunna, club-3090 disc #19 "
            "2026-05-01). Same root cause for both bugs; one fix "
            "closes both. Default ON when spec-decode active — real "
            "correctness fix, not experimental. Disable via "
            "GENESIS_DISABLE_PR37521_SPEC_DECODE_WARMUP_K=1 if K-sized "
            "warmup itself OOMs (better-than-runtime-OOM diagnosis)."
        ),
        "upstream_pr": 37521,
        "applies_to": {
            # Only fires when speculative_config is present at runtime.
            # The text-patch site itself is gated `if self.speculative_config:`
            # so non-spec-decode boots are NULL on this path.
        },
    },
    "PN32": {
        "wiring": "patch_N32_gdn_chunked_prefill",
        "title": "GDN _forward_core chunked-prefill v2 (Cliff 2 fix for single-24GB-GPU OOM)",
        "env_flag": "GENESIS_ENABLE_PN32_GDN_CHUNKED_PREFILL",
        "default_on": False,
        "category": "hybrid",
        # NOTE: v2 (v7.69) supersedes v1 (v7.65). v1 chunked at the WRONG
        # level (forward_cuda outer, didn't propagate cu_seqlens to inner
        # FLA call → empirically OOM'd EARLIER on club-3090 cross-rig).
        # v2 chunks _forward_core directly with chunk-local cu_seqlens
        # and threaded initial_state. See club-3090#19 finding 3.
        #
        # COMPOSITION: v2 chunks the OUTER FLA call (chunk_gated_delta_rule).
        # P103 chunks INSIDE chunk_gated_delta_rule_fwd's h tensor. Both
        # default OFF, COMPLEMENTARY. Recommended together for single-24GB-
        # GPU users hitting Cliff 2 (>50K single-prompt prefill on 1×3090
        # /4090/5090).
        #
        # DEPENDENCY: P28 (legacy persistent buffer pool) conflicts with
        # PN32 v2 — both modify gdn_linear_attn.py overlapping paths.
        # Operator MUST disable P28 before enabling PN32. P28 IS in this
        # dispatcher (legacy lifecycle since v7.65) — symmetric back-link
        # declared via conflicts_with below.
        "credit": (
            "Genesis-original v7.69 v2 (2026-05-02) — Cliff 2 fix per "
            "noonghunna's CLIFF2_INVESTIGATION_20260430.md + cross-rig "
            "club-3090#19 finding 3. v2 supersedes v1 (v7.65) which "
            "chunked at wrong level. v2: when single-seq prefill T > "
            "16384 (env-tunable), splits chunk_gated_delta_rule call "
            "into chunks of 8192. Each chunk: slice query/key/value/g/"
            "beta along T, build chunk-local cu_seqlens=[0, chunk_len], "
            "thread initial_state via prior chunk's last_recurrent_state, "
            "concat outputs. Multi-seq prefill bypasses to original. "
            "Default OFF. Composes with P103 (P103 chunks INSIDE FLA "
            "kernel; PN32 chunks the FLA CALL). Recommended pairing for "
            "single-24GB-GPU Cliff 2: GENESIS_ENABLE_P103=1 + GENESIS_"
            "ENABLE_PN32_GDN_CHUNKED_PREFILL=1. Cross-rig validation "
            "required (our 2×A5000 PROD with TP=2 doesn't hit Cliff 2)."
        ),
        "upstream_pr": None,
        "applies_to": {
            # Triggers in any hybrid GDN model with long single-prompts.
            # NULL on non-GDN paths (no GDN layers in 35B Qwen3MoE).
        },
        "requires_patches": [],
        "conflicts_with": ["P28"],
    },
    "PN31": {
        "wiring": "patch_N31_fa_varlen_persistent_out",
        "title": "FA varlen persistent out buffer (issue #15, sister to P38)",
        "env_flag": "GENESIS_ENABLE_PN31_FA_VARLEN_PERSISTENT_OUT",
        "default_on": False,
        "category": "memory_pool",
        "credit": (
            "Genesis-original sister patch to P38 (K_full/V_full persistent "
            "buffers). Closes issue #15 — OOM at flash_attn_varlen_func on "
            "budget-constrained single-GPU configs. Per-shape persistent "
            "out buffer eliminates per-call malloc pressure inside FA C "
            "extension. Memory cost: ~16-64 MiB per shape × layer. NULL "
            "impact on 2×A5000 PROD (we have headroom); designed for "
            "1×3090 / 1×4090 single-GPU community users."
        ),
        "upstream_pr": None,
        "applies_to": {
            # Triggers when TurboQuant attention is active. NULL on
            # non-TQ paths (FP8 KV, BF16 KV).
        },
        "conflicts_with": [],
        "requires_patches": [],
    },
    "PN30": {
        "wiring": "patch_N30_ds_layout_spec_decode_align",
        "title": "DS conv state layout + spec-decode AL>1 fix (issue #17)",
        "env_flag": "GENESIS_ENABLE_PN30_DS_LAYOUT_SPEC_DECODE",
        "default_on": False,
        "category": "model_correctness",
        "credit": (
            "Genesis-original fix for issue #17 (noonghunna, 2026-05-01). "
            "Replaces upstream NotImplementedError raise in "
            "`get_conv_copy_spec` for DS conv state layout + "
            "num_accepted_tokens > 1 (every spec-decode AL>1 prefill on "
            "DS-enabled hybrid GDN configs). Two-file text-patch: "
            "(1) mamba_utils.py uses .contiguous() copy + module-level "
            "temp tensor list; (2) v1/worker/mamba_utils.py wraps "
            "do_mamba_copy_block with stream sync + list clear after "
            "batch_memcpy. Cost: ~10-50us per batch when path active. "
            "Closes 50/50 LCB v6 failure on structured-CoT workloads."
        ),
        "upstream_pr": None,
        "applies_to": {
            # Triggers in any hybrid GDN model with DS layout + spec-decode.
            # Genesis A5000 PROD doesn't have --structured-outputs-config so
            # may not exercise this path; community single-3090 + structured
            # CoT does.
        },
        "conflicts_with": [],
        "requires_patches": [],
    },
    "P67c": {
        "wiring": "patch_67c_sparse_v",
        "title": "Per-row vote sparse-V integration into P67 split-M kernel",
        "env_flag": "GENESIS_ENABLE_P67_SPARSE_V",
        "default_on": False,
        "category": "perf_hotfix",
        "credit": (
            "Genesis-original 2026-05-01 — synthesizes PR41422 proven uniform-"
            "scalar `if` pattern (Triton 3.6 scf.if), TRT-LLM #9821 sink "
            "protection design, TheTom #41422 threshold=0 bit-exact contract. "
            "Per-q_t skip via `if SPARSE_V: ...` constexpr-DCE'd to nothing "
            "at SPARSE_V=0 → byte-equivalent to pre-sparse-V P67. "
            "When SPARSE_V=1 + threshold=0: bit-exact (P_t = exp2(...) >= 0, "
            "so `p_t_max < 0` always False). When threshold > 0: per-q_t "
            "max-prob check skips V@P tl.dot for cold tiles past sink window. "
            "Greenfield in spec-decode K+1 verify (no upstream impl exists). "
            "Expected gain: +5-22% on long-context (16K+); NULL on short ctx."
        ),
        "upstream_pr": None,
        "applies_to": {"is_turboquant": [True]},
        "requires_patches": ["P67"],
        "conflicts_with": [],
    },
    "PR41446": {
        "wiring": "patch_pr41446_gdn_chunk_o_scale_fold",
        "title": "GDN chunk_o scale-fold (vllm#41446 pattern (c))",
        "env_flag": "GENESIS_ENABLE_PR41446_GDN_SCALE_FOLD",
        "env_flag_aliases": ["GENESIS_ENABLE_PN29_GDN_SCALE_FOLD"],
        "default_on": False,
        "category": "perf_hotfix",
        "credit": (
            "Backport of vllm#41446 (zobinHuang, OPEN) pattern (c) only. "
            "Folds scale multiply in `chunk_fwd_kernel_o`: "
            "`b_o = (b_o + tl.dot(b_A, b_v)) * scale` instead of "
            "`b_o = b_o * scale + tl.dot(b_A, b_v) * scale`. "
            "One fewer fp32 multiply per inner iter. Distributive on "
            "fp32 accumulators (drift bounded by 1-2 ULP per element). "
            "Triton compiler does NOT auto-fuse across the +/- boundary, "
            "so explicit fold = guaranteed save. Hardware-agnostic; "
            "PR is MI300X-targeted but pattern (c) is NVIDIA-Triton "
            "compatible. Genesis-applicable: hybrid GDN models "
            "(Qwen3.5/3.6 27B); no-op on Qwen3MoE 35B."
        ),
        "upstream_pr": 41446,
        "applies_to": {
            # Triggers in any model using FLA chunk_fwd_kernel_o (hybrid
            # GDN). On Qwen3MoE without GDN, the kernel never fires →
            # patch is silently no-op even if env enabled.
        },
    },
    "PR41142": {
        "wiring": "patch_pr41142_gdn_a_b_contiguous",
        "title": "GDN a/b contiguity in fix_query_key_value_ordering (vllm#41142)",
        "env_flag": "GENESIS_ENABLE_PR41142_GDN_AB_CONTIGUOUS",
        "env_flag_aliases": ["GENESIS_ENABLE_PN11_GDN_AB_CONTIGUOUS"],
        "default_on": False,
        "category": "model_correctness",
        "credit": (
            "Backport of vllm#41142 (Yeuvoir, OPEN). Fixes upstream issue "
            "#41112: in `GatedDeltaNetAttention.fix_query_key_value_ordering` "
            "the reshape of `b` and `a` returns a non-contiguous view when "
            "num_v_heads == num_k_heads (np/ng == 1), breaking "
            "`fused_post_conv_prep` Triton kernel which assumes head-dim "
            "stride 1. Adds `.contiguous()` to both lines (zero cost when "
            "already contiguous; copy only on the buggy path). Symptom on "
            "affected configs: silent quality drift, no crash. For Genesis "
            "prod (Qwen3.6 27B has np/ng=8, 35B has no GDN) this is "
            "DEFENSIVE — installs guard against future model swaps."
        ),
        "upstream_pr": 41142,
        "applies_to": {
            # Patch only matters when GDN layer's fix_query_key_value_ordering
            # runs with np/ng==1. Genesis prod doesn't trigger it but the
            # patch is harmless (no-op .contiguous() call).
        },
    },
    "PR34207": {
        "wiring": "patch_pr34207_ffn_intermediate_pool",
        "title": "FFN intermediate scratch pool — Cliff 1 fix on TQ3 path",
        "env_flag": "GENESIS_ENABLE_PR34207_FFN_INTERMEDIATE_POOL",
        "env_flag_aliases": ["GENESIS_ENABLE_PN12_FFN_INTERMEDIATE_POOL"],
        "default_on": False,
        "category": "memory_savings",
        "credit": (
            "Genesis-original 2026-04-29 — Cliff 1 fix on TQ3 path. "
            "Closes 138 MiB OOM at 192K + tool-call on RTX 3090 (noonghunna "
            "report). PR40849 closes Cliff 1 on FP8 by freeing ~600 MiB persistent "
            "draft VRAM, but on TQ3 frees only ~230 MiB — not enough slack "
            "for the 138 MiB transient. Different memory class. PR34207 pools "
            "the SiluAndMul output across layers (single buffer per "
            "(intermediate_size, dtype, device)) — reduces per-step allocator "
            "churn from ~4.7-18 GiB to ~73-285 MiB on Lorbus 27B-int4. "
            "Pointer-stable (cudagraph-safe). Cross-engine reference: "
            "TensorRT-LLM live-range activation reuse (gold standard); "
            "alternative paths: vLLM PR #34207 (silu_and_mul.out variant), "
            "SGLang PR #15927 (piecewise CUDA graph private pool). Tested "
            "via 17 unit tests in tests/test_ffn_intermediate_cache.py."
        ),
        "upstream_pr": 34207,  # would obsolete this patch if merged
        "applies_to": {
            # Patch matters when SiluAndMul / MulAndSilu is on the hot path
            # (any model with FFN gate-up + silu activation — qwen3, llama,
            # mistral, deepseek, etc.). For MoE models impact is per-expert.
        },
    },
    "PR41268": {
        "wiring": "patch_pr41268_scoped_max_split",
        "title": "Scoped max_split_size_mb during model load (vllm#41268)",
        "env_flag": "GENESIS_ENABLE_PR41268_SCOPED_MAX_SPLIT",
        "env_flag_aliases": ["GENESIS_ENABLE_PN19_SCOPED_MAX_SPLIT"],
        "default_on": False,
        "category": "memory_savings",
        "credit": (
            "Backport of vllm#41268 (MatthewBonanni, OPEN 2026-04-30). "
            "PyTorch 2.10+ introduced load-time allocator fragmentation: "
            "weight segments split inside other segments, leaving "
            "200-500 MiB unusable. Mitigation: temporarily set "
            "max_split_size_mb=20 (PyTorch minimum) for the duration of "
            "model load, restore prior on exit. Cudagraph-safe (load-"
            "time only; capture phase uses restored allocator). "
            "Default OFF — operator should measure fragmentation gap "
            "via nvidia-smi peak during load before vs after to confirm "
            "win on Ampere SM 8.6 (PR #41268 measured on H100; A5000 "
            "behavior unverified). Cross-reference Genesis memory "
            "feedback_p104_l2_persistence_thrashing — hardware-mismatch "
            "patches are anti-pattern; measure first."
        ),
        "upstream_pr": 41268,
        "applies_to": {
            # Always applicable on CUDA. Self-detects torch < 2.11 lack
            # of _accelerator_setAllocatorSettings and falls through
            # unchanged.
        },
    },
    "PR40898": {
        "wiring": "patch_pr40898_dflash_swa_support",
        "title": "DFlash SWA support partial backport (vllm#40898)",
        "env_flag": "GENESIS_ENABLE_PR40898_DFLASH_SWA",
        "env_flag_aliases": ["GENESIS_ENABLE_PN21_DFLASH_SWA"],
        "default_on": False,
        "category": "spec_decode",
        "credit": (
            "Partial backport of vllm#40898 (jianc99, OPEN 2026-05-01). "
            "Adds SWA config preservation in speculators/algos.py and forces "
            "causal=True on sliding-window layer attention metadata in "
            "v1/spec_decode/dflash.py. The qwen3_dflash.py model class "
            "changes (7+ sub-patches) are NOT backported. EMPIRICAL on 35B-A3B "
            "DFlash 160K: tool-call regresses 5-6/7 vs 7/7 baseline (without PR40898) — "
            "metadata/compute mismatch (config says SWA, model computes full attn). "
            "DEFAULT OFF, NOT enabled in any launch script. Wait for upstream merge "
            "or full manual model class backport before enabling. Composes (no conflict) "
            "with PR40727 if/when full enabler lands."
        ),
        "upstream_pr": 40898,
        "applies_to": {},
        "conflicts_with": [],
        "requires_patches": [],  # Pairs with PR40727 but does not strictly require it
    },
    "PR39419": {
        "wiring": "patch_pr39419_local_argmax_tp",
        "title": "Local argmax for TP draft (vllm#39419 backport)",
        "env_flag": "GENESIS_ENABLE_PR39419_LOCAL_ARGMAX_TP",
        "env_flag_aliases": ["GENESIS_ENABLE_PN22_LOCAL_ARGMAX_TP"],
        "default_on": False,
        "category": "spec_decode",
        "credit": (
            "Backport of vllm#39419 (EanWang, OPEN 2026-05-01). Adds "
            "get_top_tokens() plumbing to Qwen3 and Qwen3-DFlash model "
            "classes, enabling vocab-parallel argmax on each TP rank "
            "instead of all-gathering full logits. Wins +9.4-30.6% TPS "
            "on TP>=2 + draft model per PR author. LogitsProcessor."
            "get_top_tokens() callsite is already in our pin (PR #34049 "
            "merged). Llama and Eagle3 parts of the upstream PR are not "
            "backported — Genesis does not run those models in production."
        ),
        "upstream_pr": 39419,
        "applies_to": {},
        "conflicts_with": [],
        "requires_patches": [],
    },
    "PR40727": {
        "wiring": "patch_pr40727_dflash_aux_layer_indexing",
        "title": "DFlash aux layer +1 indexing fix (vllm#40727)",
        "env_flag": "GENESIS_ENABLE_PR40727_DFLASH_AUX_LAYER_FIX",
        "env_flag_aliases": ["GENESIS_ENABLE_PN24_DFLASH_AUX_LAYER_FIX"],
        "default_on": False,
        "category": "spec_decode",
        "credit": (
            "Backport of vllm#40727 (benchislett, OPEN 2026-05-01). One-line "
            "semantic fix in _get_eagle3_aux_layers_from_config. DFlash stores "
            "target_layer_ids as 0-indexed; downstream Eagle3 aux machinery "
            "expects 1-indexed (layer 0 = embedding). +1 shift converts. "
            "Empirical: AL gsm8k 6.18→6.42 per PR author."
        ),
        "upstream_pr": 40727,
        "applies_to": {},
        "conflicts_with": [],
        "requires_patches": [],
    },
    "PR39148": {
        "wiring": "patch_pr39148_merge_attn_states_nan_guard",
        "title": "merge_attn_states NaN guard (vllm#39148 backport)",
        "env_flag": "GENESIS_ENABLE_PR39148_MERGE_ATTN_NAN_GUARD",
        "env_flag_aliases": ["GENESIS_ENABLE_PN28_MERGE_ATTN_NAN_GUARD"],
        "default_on": False,
        "category": "perf_hotfix",
        "credit": (
            "Backport of vllm#39148 (jasonkim8652, OPEN 2026-05-01). "
            "Branchless NaN guard in Triton merge_attn_states kernel for "
            "both-LSE-(-inf) edge case (zero-context-length chunked prefill). "
            "Without guard: NaN propagates through exp()/division and silently "
            "corrupts output — one bad token can break tool-call JSON parsing. "
            "Fix: clamp max_lse to -1e30 finite floor + add 1e-10 epsilon to "
            "denominator. Quality-only — no perf impact. CUDA merge_attn_states "
            "kernel already had this guard; PR39148 brings Triton to parity."
        ),
        "upstream_pr": 39148,
        "applies_to": {},
        "conflicts_with": [],
        "requires_patches": [],
    },
    "P15B": {
        "wiring": "patch_15B_fa_varlen_clamp",
        "title": "FA varlen max_seqlen_k clamp on TQ path (Issue #15 fix)",
        "env_flag": "GENESIS_ENABLE_P15B_FA_VARLEN_CLAMP",
        "default_on": False,
        "category": "perf_hotfix",
        "credit": (
            "Genesis-original 2026-05-01 fix for noonghunna's Issue #15. "
            "PN17 clamps max_seqlen_k on the FA2 backend path, but TurboQuant "
            "code path bypasses PN17's coverage by calling vllm_flash_attn's "
            "vendored wrapper via turboquant_attn.py:_flash_attn_varlen. P15B "
            "extends the same clamp logic to that callsite via text-patch — "
            "computes actual span from cu_seqlens_k and clamps max_seqlen_k "
            "before invocation. Prevents 50 MiB workspace OOM on long-context "
            "continuation-prefill on tight VRAM (24 GB consumer cards). "
            "Trade-off: adds one GPU->CPU sync per call on the infrequent "
            "continuation-prefill path."
        ),
        "upstream_pr": None,
        "applies_to": {},
        "conflicts_with": [],
        "requires_patches": [],
    },
    "P38B": {
        "wiring": "patch_38b_compile_safe_hook",
        "title": "P38 compile-safe in-source hook (Issue #14 fix)",
        "env_flag": "GENESIS_ENABLE_P38B_COMPILE_SAFE",
        "default_on": False,
        "category": "perf_hotfix",
        "credit": (
            "Genesis-original 2026-05-01 fix for noonghunna's Issue #14. "
            "Root cause: aot_compile_fullgraph captures _continuation_prefill "
            "original body at engine init; Python class-attribute rebind "
            "(P38's mechanism) doesn't propagate to compiled artifact. "
            "P38B injects an in-source delegate hook at the start of "
            "_continuation_prefill body via text-patch. Hook calls a "
            "dispatcher that returns Genesis result OR None (fall-through). "
            "Source-level edit means aot_compile captures the hook itself. "
            "Affects ALL TQ KV users with V0/V1 compile pipeline; fp8 KV "
            "configs unaffected (different code path). Composes with P38 "
            "(both share _genesis_continuation_prefill impl)."
        ),
        "upstream_pr": None,
        "applies_to": {},
        "conflicts_with": [],
        "requires_patches": [],  # P38 install order: P38 first (provides impl), P38B second (installs hook)
    },
    "PR41422": {
        "wiring": "patch_pr41418_sparse_v_kernel",
        "title": "Sparse-V tile-skip Genesis kernel (BLASST λ=a/L for SM86)",
        "env_flag": "GENESIS_ENABLE_PR41422_SPARSE_V",
        "env_flag_aliases": ["GENESIS_ENABLE_PN26_SPARSE_V"],
        "default_on": False,
        "category": "perf_hotfix",
        "credit": (
            "Genesis-original Triton kernel fork — first sparse-V tile-skip "
            "deployed for SM86 (Ampere consumer). Synthesized from 4-agent "
            "research 2026-05-01: vllm#41422 (TheTom, AMD-only validated) "
            "design template + BLASST arXiv 2512.12087 (Yuan et al. Dec 2025) "
            "λ=a/L threshold formula + tq-kv reference (CUDA, SM86-compatible) "
            "acc*re_scale skip semantics + StreamingLLM (arXiv 2309.17453) "
            "sink token protection (first 4 KV positions never skipped). "
            "Mechanism: when tl.max(p) < threshold for a KV tile, skip V load + "
            "dequant + weighted sum, just decay accumulator. Online softmax "
            "denominator/max still update so totals stay numerically exact "
            "for non-skipped tiles. Composes with PR41418 main (centroids "
            "prebake) + PR40941 (workspace revert) + P67 (multi-query — separate "
            "code path, not affected). Default OFF; opt-in via "
            "GENESIS_ENABLE_PR41422_SPARSE_V=1 + GENESIS_PR41418_SPARSE_V_THRESHOLD "
            "(fixed) OR GENESIS_PR41418_SPARSE_V_SCALE_FACTOR (BLASST adaptive)."
        ),
        "upstream_pr": 41422,
        "applies_to": {},
        "conflicts_with": [],
        "requires_patches": [],
    },
    "PR41418": {
        "wiring": "patch_pr41418_tq_unified_perf",
        "title": "TQ unified perf pack (centroids prebake + sparse V scaffold)",
        "env_flag": "GENESIS_ENABLE_PR41418_TQ_UNIFIED",
        "env_flag_aliases": ["GENESIS_ENABLE_PN26_TQ_UNIFIED"],
        "default_on": False,
        "category": "perf_hotfix",
        "credit": (
            "Genesis-original 2026-05-01 unification of three OPEN upstream "
            "PRs (jasonkim8652): #41418 pre-baked Lloyd-Max centroids (drop-in "
            "safe, eliminates 50ms-2.5s JIT solver per shape on cold boot); "
            "#41422 sparse V tile-skip in decode kernel (scaffolded, OFF by "
            "default until NVIDIA Ampere correctness validation — author "
            "validated AMD MI300X only); #41414 head_dim pow-2 padding "
            "DROPPED — Qwen3.6 head_dim=128 already pow-2, would add dead "
            "code overhead. Genesis defensive addition: self-check at "
            "module-init asserts prebaked centroids equal solver output; on "
            "drift (e.g. upstream changes Lloyd-Max algo) auto-disables "
            "prebake and falls through to runtime solver with WARNING. No "
            "silent staleness. Composes with P67/PR40941/PR40849 — orthogonal code "
            "paths."
        ),
        "upstream_pr": 41418,
        "applies_to": {},
        "conflicts_with": [],
        "requires_patches": [],
    },
    "PN25": {
        "wiring": "patch_N25_silu_inductor_safe_pool",
        "title": "SiluAndMul.forward_native opaque-op pool (Cliff 1 mech B compile path)",
        "env_flag": "GENESIS_ENABLE_PN25_SILU_INDUCTOR_SAFE",
        "default_on": False,
        "category": "memory_savings",
        "credit": (
            "Genesis-original 2026-05-01 in response to noonghunna's "
            "club-3090#16 (VolandBerlioz/ampersandru cross-rig OOM trace, "
            "RTX 3090 24 GB + Lorbus 27B + OpenCode 29K prefill). PR34207 "
            "patches eager `forward_cuda` but `custom_ops=['none']` (default "
            "under V1 aot_compile_fullgraph) routes dispatch through "
            "`forward_native` which Inductor inlines and lowers to "
            "`empty_strided_cuda(...)`, bypassing PR34207's pool. "
            "Sister-patch PN25 patches `forward_native` to dispatch through "
            "an opaque `genesis::silu_and_mul_pooled` torch.library.custom_op "
            "(Inductor cannot inline opaque ops). Both patches share the "
            "same FFNIntermediateCache pool. Recommended pairing for any "
            "inductor-heavy config."
        ),
        "upstream_pr": None,
        "applies_to": {},
        "conflicts_with": [],
        "requires_patches": [],  # complements PR34207 but does not require it
    },
    "PN17": {
        "wiring": "patch_N17_fa2_softmax_lse_clamp",
        "title": "FA2 softmax_lse runtime clamp (Cliff 1 mechanism A, Issue #11)",
        "env_flag": "GENESIS_ENABLE_PN17_FA2_LSE_CLAMP",
        "default_on": False,
        "category": "memory_savings",
        "credit": (
            "Genesis-original 2026-04-30 in response to noonghunna's "
            "Genesis Issue #11 cross-rig diagnosis (RTX 3090, 2026-04-29). "
            "FA2 `flash_attn_varlen_func` allocates softmax_lse buffer "
            "of shape [num_seqs, num_heads, max_seqlen_k] sized by "
            "the max_seqlen_k argument — NOT actual seqused_k. vLLM's "
            "gpu_model_runner sets attn_metadata.max_seq_len = "
            "max_model_len during cudagraph capture for shape stability "
            "(see vllm#40961 SWA case); this leaks into runtime "
            "decode/prefill, causing 50-100 MiB over-allocation at "
            "long context. Closes Cliff 1 mechanism A (FA2 path); "
            "widens long-text-no-vision safe envelope from ~150K to "
            "~205K. Mechanism B (FFN intermediate buffer 138 MiB on "
            "long-vision) is OUT OF SCOPE — requires upstream-FFN "
            "chunked forward, not addressable from Genesis text-patch "
            "layer. Cudagraph-safe: clamp only fires when "
            "is_current_stream_capturing() returns False; capture-time "
            "preserves max_model_len padding. Reference: "
            "Dao-AILab/flash-attention#1011 (open since 2024)."
        ),
        "upstream_pr": None,
        "applies_to": {
            # Applies whenever FA2 varlen path is active. Most relevant
            # at long context (>100K) where the cap-leak dominates.
        },
    },
    "PN16": {
        "wiring": "patch_N16_lazy_reasoner",
        "title": "Lazy-reasoner request hook (per-request enable_thinking)",
        "env_flag": "GENESIS_ENABLE_PN16_LAZY_REASONER",
        "default_on": False,
        "category": "request_middleware",
        "credit": (
            "Genesis-original 2026-04-29. Hybrid policy on whether the "
            "model's `<think>...</think>` reasoning block adds value for "
            "this specific request. Variant 1 (pre-decision): force "
            "enable_thinking=False on short prompts (< "
            "GENESIS_PN16_THRESHOLD_CHARS, default 300) without tools, "
            "json_schema, or reasoning-signal patterns (math/code/CoT "
            "keywords). Variant 3 (client override): respect explicit "
            "chat_template_kwargs.enable_thinking from the client. Variant "
            "4 (LogitsProcessor cap): UPSTREAM-BLOCKED — vllm v1 rejects "
            "custom logits processors when speculative_config is set "
            "(Genesis PROD = MTP K=3). Variant 5 (prompt-engineering soft "
            "cap): fallback when max_thinking_tokens > 0; appends a "
            "concise-reasoning hint to the last user message. Soft cap, "
            "depends on model compliance, but works with spec-decode. "
            "Goal: reduce wasted reasoning tokens + TTFT on simple prompts "
            "without doubling latency or load. Stats counters exposed via "
            "`vllm._genesis.middleware.lazy_reasoner.get_stats()`."
        ),
        "upstream_pr": None,
    },
    "PR40074": {
        "wiring": "patch_pr40074_tq_decode_oob_clamp",
        "title": "TQ decode IOOB safe_page_idx clamp (vllm#40074)",
        "env_flag": "GENESIS_ENABLE_PR40074_TQ_DECODE_OOB_CLAMP",
        "env_flag_aliases": ["GENESIS_ENABLE_PN14_TQ_DECODE_OOB_CLAMP"],
        "default_on": False,
        "category": "kernel_safety",
        "credit": (
            "Backport of vllm#40074 (devarakondasrikanth @adobe, OPEN). "
            "Fixes upstream issue #39998 — Triton bounds-checker assertion "
            "in `_tq_decode_stage1` on long (>32k) sequences. The mask= "
            "argument guards the LOADED VALUE on masked-out lanes but not "
            "the address arithmetic; clamping page_idx to 0 via "
            "`tl.where(kv_mask, page_idx, 0)` keeps the pointer in-bounds "
            "even on lanes whose result is discarded. Originally reported "
            "on 4090 (sm_89); jhsmith409 confirmed clean apply on 5090 "
            "(sm_120) while stacking on top of #39931. Defensive on Genesis "
            "Ampere prod (sm_86 — assertion not seen). Becomes load-bearing "
            "on Sander's planned RTX PRO 6000 Blackwell upgrade. Self-"
            "retires via marker `safe_page_idx` when #40074 merges. "
            "Codepath fires when spec-decode OFF/K=1 OR P67 dispatch returns "
            "False (shape outside envelope) — runs in Genesis prod despite "
            "MTP K=3 being active."
        ),
        "upstream_pr": 40074,
        "applies_to": {
            "is_turboquant": [True],
        },
    },
    # PR41235 entry moved to legacy/retired section below (lifecycle: retired_2026-05-04)
    # Reason: vllm 0.20.2 commit c2fb013 merged identical change (#41235).
    # See PR41235 entry near line 1289 for retirement metadata.

    "PR41043": {
        "wiring": "patch_pr41043_spec_decode_zero_alloc",
        "title": "Spec-decode prepare_next_token_ids_padded zero-alloc (vllm#41043)",
        "env_flag": "GENESIS_ENABLE_PR41043_SPEC_PREPARE_NEXT_IDS_ZERO_ALLOC",
        "env_flag_aliases": ["GENESIS_ENABLE_P94"],
        "default_on": False,
        "category": "spec_decode",
        "credit": "Backport of vllm#41043 (wangluochao902, MERGED 2026-04-29). Removes GPU->CPU .tolist() sync + list-comp Python objects + np.array allocation in LLMBaseProposer.prepare_next_token_ids_padded hot path. PR author measured PR40941b TPOT -9.3% on Llama-3.1-8B + Eagle3 TP=4. For our MTP K=3 single-stream: expected +2-4% wall TPS + tighter CV. SUPERSEDED-ON-MERGE: when our pin advances past the merge SHA the patch will SKIP cleanly via drift detection on the original .tolist() anchor — at that point delete the wiring file + this entry.",
        "upstream_pr": 41043,
        "superseded_by": "vllm#41043 (merged 2026-04-29)",
        "applies_to": {
            # Applies whenever spec-decode is active. All spec methods.
        },
    },
    "PR41127": {
        "wiring": "patch_pr41127_flashinfer_full_cg_specdec",
        "title": "FlashInfer FULL CUDA graph for spec-decode (vllm#41127)",
        "env_flag": "GENESIS_ENABLE_PR41127_FLASHINFER_FULL_CUDAGRAPH",
        "env_flag_aliases": ["GENESIS_ENABLE_P100"],
        "default_on": False,
        "category": "perf_hotfix",
        "credit": "Backport of vllm#41127 (open 2026-04-28). Per Sander 'не ждём, изучаем, импортируем'. Native FlashInfer can route uniform query_len>1 (1+num_spec_tokens) batches through prefill wrapper in cudagraph mode (zero_rows padding bit-identical). Adds FISpecDecode dataclass + _get_spec_decode_prefill_wrapper method + per-row qo_indptr delta scan in build() + FISpecDecode case in forward(). 11 sub-patches on flashinfer.py. NO-OP for PROD (turboquant_attn). Active for 27B variants (FlashInfer + spec-decode + non-DCP). Expected: +5-10% TPS on Ampere SM 8.6. RECOMMENDED on Blackwell consumer (sm_120) where FlashInfer is the default backend and PIECEWISE downgrade was observed (apnar club-3090#51). Recommendation surfaced via gpu_profile.PATCH_RECOMMENDATIONS rule.",
        "upstream_pr": 41127,
        "applies_to": {},  # FlashInfer auto-selected; gating via env_flag only
    },
    "P103": {
        "wiring": "patch_103_fla_cliff2_chunked",
        "title": "FLA Cliff 2 chunked fwd_h+fwd_o orchestrator (qwen36-27b-single-3090#1)",
        "env_flag": "GENESIS_ENABLE_P103",
        "default_on": False,
        "category": "memory_hotfix",
        "credit": "Genesis-original 2026-04-28 in response to noonghunna Cliff 2 OOM report (qwen36-27b-single-3090#1). Wraps chunk.py::chunk_gated_delta_rule_fwd to split T-dim into MAX_T sub-prompts; runs fwd_h + fwd_o per sub-call, chains final_state, never materializes full (B, NT, H, V, K) h tensor. For Qwen3.6-27B at T=64K: peak h drops 4x (805 → 200 MiB per rank). Saves ~600 MiB headroom for long-context single-GPU users. Falls back to original for cu_seqlens != None or T <= MAX_T. Default OFF; opt-in via GENESIS_ENABLE_P103=1. Threshold: GENESIS_FLA_FWD_H_MAX_T (default 16384, rounded down to FLA_CHUNK_SIZE multiple). KDA path uncovered (separate model class).",
        "upstream_pr": None,
        "applies_to": {
            "model_arch": [
                "Qwen3MoeForCausalLM",
                "Qwen3_5ForConditionalGeneration",
                "Qwen3NextForCausalLM",
            ],
        },
    },
    "PR41123": {
        "wiring": "patch_pr41123_tq_continuation_slicing",
        "title": "TQ continuation 64-token slicing (vllm#41123 SELECTIVE)",
        "env_flag": "GENESIS_ENABLE_PR41123_TQ_CONTINUATION_64TOK_SLICE",
        "env_flag_aliases": ["GENESIS_ENABLE_P101"],
        "default_on": False,
        "category": "perf_hotfix",
        "credit": "Selective backport of vllm#41123 TQ on hybrid models. TAKE: _CONTINUATION_DECODE_THRESHOLD 128→64 + _CONTINUATION_DECODE_MAX_CACHED_LEN=32K + 64-token slicing loop in _prefill_attention. SKIP: cudagraph_support downgrade (would hurt PROD), hybrid boundary-skip (would break our explicit skip-layers). Expected: +3-12% TPS on PROD long-context. Composes with PR40941/PR40941b.",
        "upstream_pr": 41123,
        "applies_to": {
            "kv_cache_dtype": [
                "turboquant_k8v4", "turboquant_4bit_nc",
                "turboquant_k3v4_nc", "turboquant_3bit_nc",
            ],
        },
    },
    "PR40941b": {
        "wiring": "patch_pr40941b_workspace_manager_memoize",
        "title": "WorkspaceManager.get_simultaneous memoization (perf hotfix)",
        "env_flag": "GENESIS_ENABLE_PR40941B_WORKSPACE_GET_SIMULTANEOUS_MEMO",
        "env_flag_aliases": ["GENESIS_ENABLE_P99"],
        "default_on": False,
        "category": "perf_hotfix",
        "credit": "Per Sander 2026-04-28: 'if revert gives speedup, look at kernel — maybe rewrite'. PR40941b keeps upstream WorkspaceManager design (shared memory, 60x savings) but adds memoization to bypass per-call list-comp + accumulate + _ensure_workspace_size. Cache hit ~5x faster than full computation. Composes with PR40941 (PR40941 reverts turboquant_attn to per-layer; PR40941b helps any other backend using WorkspaceManager).",
        "upstream_pr": 40941,
        "applies_to": {},  # applies whenever WorkspaceManager is used
    },
    "PR40941": {
        "wiring": "patch_pr40941_tq_workspace_revert",
        "title": "TQ WorkspaceManager revert (vllm#40941 perf hotfix)",
        "env_flag": "GENESIS_ENABLE_PR40941_TQ_WORKSPACE_REVERT",
        "env_flag_aliases": ["GENESIS_ENABLE_P98"],
        "default_on": False,
        "category": "perf_hotfix",
        "credit": "Reverts upstream PR #40941 (MERGED 2026-04-27). PR introduced WorkspaceManager indirection in turboquant_attn._decode_attention hot path. Diagnosis 2026-04-28: caused 17% TPS regression on PROD (200 → 167 TPS) due to current_workspace_manager().get_simultaneous() Python lookup × N layers × per-step. Restores OLD per-layer cached buffer pattern. Memory cost: O(num_layers) extra dequant buffers (~1GB for 64-layer model). DO NOT enable on H100/H200 high-concurrency where WorkspaceManager amortizes better. NOTE: this patch is a DELIBERATE INVERSE of merged upstream behavior (NOT a backport) — it remains a perf hotfix specifically for Ampere small-batch single-stream workloads even though the upstream PR is merged. Author: Sandermage.",
        "upstream_pr": 40941,
        "applies_to": {
            "kv_cache_dtype": [
                "turboquant_k8v4", "turboquant_4bit_nc",
                "turboquant_k3v4_nc", "turboquant_3bit_nc",
            ],
        },
    },
    "PR40385": {
        "wiring": "patch_pr40385_marlin_tp_cudagraph_cap",
        "title": "Marlin TP cudagraph cap on Ampere (vllm#40385)",
        "env_flag": "GENESIS_ENABLE_PR40385_MARLIN_TP_CUDAGRAPH_CAP",
        "env_flag_aliases": ["GENESIS_ENABLE_P95"],
        "default_on": False,
        "category": "stability",
        "credit": "Backport of vllm#40385 (OPEN as of 2026-04-28). Defensive cap of max_cudagraph_capture_size to 8 when ALL of: TP>1, Ampere SM 8.0 family (covers SM 8.6 A5000), quantization endswith '_marlin', AND user did NOT set explicit cudagraph sizing. Mitigates vllm#40121 (illegal memory access during CG replay on TP>1 + Marlin + Ampere). NO-OP for our PROD (FP8, not Marlin); ACTIVE for Lorbus INT4 + Minachist gs128 (Marlin path). Operator override via --compilation-config bypasses entirely.",
        "upstream_pr": 40385,
        "applies_to": {
            "quant_format": [
                "gptq_int4", "gptq_int8", "awq_int4", "awq_int8",
                "compressed_tensors", "int4_w4a16", "int8_w8a16",
                "autoround_int4", "autoround_int8",
            ],
        },
    },

    # ─── Legacy patches (P1–P46 series, pre-dispatcher era) ─────────────
    # These patches predate the PATCH_REGISTRY metadata system. They have
    # been live in PROD since pre-v7.0 and don't currently read an env
    # flag — they apply unconditionally as part of `apply_all`. The
    # synthetic `GENESIS_LEGACY_P*` env_flags below exist purely so the
    # dispatcher / validator / `genesis explain` see a coherent registry
    # entry; setting them has no runtime effect (yet). Future work: wire
    # actual opt-out gating where it makes sense.
    #
    # Why register them: lets `apply_all_dispatcher_sync` test pass,
    # surfaces these patches in `genesis list-patches` / `genesis explain`,
    # and provides a stable shape for documentation tooling.

    "P1": {
        "title": "FP8 kernel dispatcher (P1/P2 — Ampere FP8 viability)",
        "env_flag": "GENESIS_LEGACY_P1",
        "default_on": True,
        "lifecycle": "legacy",
        "category": "quantization",
        "credit": "Pre-dispatcher legacy patch. Wires Ampere SM86 to FP8 kernel paths so consumer 3090/A5000 can serve FP8-quantized models.",
    },
    "P3": {
        "wiring": "patch_3_tq_bf16_cast",
        "title": "TurboQuant BF16→FP8 cast (Ampere fix)",
        "env_flag": "GENESIS_LEGACY_P3",
        "default_on": True,
        "lifecycle": "legacy",
        "category": "quantization",
        "credit": "Pre-dispatcher legacy patch. Inserts BF16→FP8 cast on TQ ingress for SM86 where FP8 is software-emulated.",
    },
    "P4": {
        "wiring": "patch_4_tq_hybrid",
        "title": "TurboQuant hybrid model support",
        "env_flag": "GENESIS_LEGACY_P4",
        "default_on": True,
        "lifecycle": "legacy",
        "category": "kv_cache",
        "credit": "Pre-dispatcher legacy patch. Removes hybrid (GDN + full attention) model rejection in TQ path, enabling Qwen3.5/3.6 hybrid serving with TQ k8v4.",
        # 2026-05-05: SUPERSEDED upstream by vllm#39931 (MERGED 2026-05-05 00:14
        # UTC, JartX + jhsmith409 + Sandermage co-authors). Upstream now
        # detects hybrid via layer_types/layers_block_type/attn_type_list and
        # computes TQ page-size via lcm in `_align_hybrid_block_size` —
        # cleaner than P4. Plan: retire P4 on next pin bump past commit
        # 4f2af1a7c03aae2b3227dd7e69d726104d44a711. Verify hybrid TQ smoke test
        # boots cleanly with P4 OFF before final retirement.
        "superseded_by": "vllm#39931 (merged 2026-05-05)",
        "retire_after_pin": "0.20.2rc1+",
    },
    "P5": {
        "wiring": "patch_5_page_size",
        "title": "KV cache page size unification",
        "env_flag": "GENESIS_LEGACY_P5",
        "default_on": True,
        "lifecycle": "legacy",
        "category": "kv_cache",
        "credit": "Pre-dispatcher legacy patch. Unifies per-layer page size across hybrid attention layers so block manager doesn't fragment.",
    },
    "P5b": {
        "title": "KV page-size pad-smaller-to-max (env opt-in)",
        # Audit P2 fix 2026-05-05: registry was `GENESIS_ENABLE_P5B_PAGE_PAD`
        # but wiring code + docstrings use `GENESIS_ENABLE_P5B`. Aligned.
        "env_flag": "GENESIS_ENABLE_P5B",
        "default_on": False,
        "lifecycle": "legacy",
        "category": "kv_cache",
        "credit": "Pre-dispatcher legacy patch. Opt-in companion to P5 — pads smaller pages up to max so all layers share one block-pool stride. Guarded by env (was always opt-in).",
    },
    "P6": {
        "wiring": "patch_6_tq_block_size_align",
        "title": "TurboQuant-aware attention page size",
        "env_flag": "GENESIS_LEGACY_P6",
        "default_on": True,
        "lifecycle": "legacy",
        "category": "kv_cache",
        "credit": "Pre-dispatcher legacy patch. Selects TQ-aware page size (matches TQ packed slot stride) when TQ KV is active.",
    },
    "P7": {
        "title": "GDN dual-stream in_proj parallelism",
        "env_flag": "GENESIS_LEGACY_P7",
        "default_on": True,
        "lifecycle": "legacy",
        "category": "kernel_perf",
        "credit": "Pre-dispatcher legacy patch. Splits GDN in_proj across two CUDA streams so q/k/v projections overlap. Validated +8% decode on 35B.",
        "conflicts_with": ["P7b"],
    },
    "P7b": {
        "title": "GDN dual-stream via torch.library.custom_op (opt-in)",
        # Audit P2 fix 2026-05-05: registry was `GENESIS_ENABLE_P7B_DUAL_STREAM_CUSTOM_OP`
        # but wiring code + docstrings use `GENESIS_ENABLE_P7B`. Aligned.
        "env_flag": "GENESIS_ENABLE_P7B",
        "default_on": False,
        "lifecycle": "legacy",
        "category": "kernel_perf",
        "credit": "Pre-dispatcher legacy patch. Custom-op variant of P7 dual-stream — opt-in alternative for cudagraph capture compatibility experiments.",
        "conflicts_with": ["P7"],
    },
    "PR41235": {
        "wiring": "patch_pr41235_cuda_graph_lambda_arity",
        "title": "CUDAGraphWrapper lambda arity (vllm#41235 backport) — RETIRED 2026-05-04",
        "env_flag": "GENESIS_ENABLE_PR41235_CUDA_GRAPH_LAMBDA_ARITY",
        "env_flag_aliases": ["GENESIS_ENABLE_PN13_CUDA_GRAPH_LAMBDA_ARITY"],
        "default_on": False,
        "lifecycle": "retired",
        "notes": (
            "upstream_native_via_pr41235 — vllm 0.20.2 (commit "
            "c2fb013) merged identical change in cuda_graph.py: "
            "patch(\"gc.collect\", lambda *args, **kwargs: None) + "
            "patch(\"torch.accelerator.empty_cache\", lambda *args, "
            "**kwargs: None). Upstream code now matches our PR41235 "
            "replacement byte-for-byte. PR41235 anchor (pre-fix lambda: "
            "None pattern) no longer matches → silent skip. Per Sander "
            "rule (2026-05-04): «если код соответствует патчу, патч "
            "отключаем». Retired."
        ),
        "category": "compile_safety",
        "credit": "Backport of vllm#41235 by Roi Koren (NVIDIA). RETIRED — upstream natively fixes after vllm v0.20.2.",
        "upstream_pr": 41235,
    },
    "P8": {
        "title": "KV hybrid reporting (per-token capacity) — RETIRED 2026-05-04",
        "env_flag": "GENESIS_LEGACY_P8",
        "default_on": False,
        "lifecycle": "retired",
        "notes": (
            "upstream_native_via_get_max_concurrency_refactor — vllm "
            "v0.20.2rc1.dev9 (commit 01d4d1ad3) refactored "
            "_report_kv_cache_config to use "
            "get_max_concurrency_for_kv_cache_config(vllm_config, "
            "kv_cache_config) which natively handles hybrid layouts "
            "(SWA / chunked-local groups with per-request block count "
            "capped by window). The new formula `max_concurrency * "
            "max_model_len` supersedes our P8 approach (excluding O(1) "
            "Mamba groups from per-token divisor). Engine now reports "
            "correct capacity natively without our patch. P8 anchors "
            "no longer match (kv_cache_utils.py refactored)."
        ),
        "category": "kv_cache",
        "credit": "Pre-dispatcher legacy patch. Reports KV capacity per-token (not per-block) for hybrid models so scheduler doesn't over-admit. RETIRED upstream natively fixes after vllm v0.20.2.",
    },
    "P12": {
        "wiring": "patch_12_tool_call_reasoning",
        "title": "Qwen3 <tool_call> implicit reasoning end",
        "env_flag": "GENESIS_LEGACY_P12",
        "default_on": True,
        "lifecycle": "legacy",
        "category": "structured_output",
        "credit": "Pre-dispatcher legacy patch. Treats <tool_call> emission as implicit </think>, fixing Qwen3 reasoning models that omit explicit </think> before tool calls. Updated v7.62.5 to FIRST-occurrence (was LAST), retiring P61.",
    },
    "P14": {
        "title": "block_table tail zero-fill",
        "env_flag": "GENESIS_LEGACY_P14",
        "default_on": True,
        "lifecycle": "legacy",
        "category": "kernel_safety",
        "credit": "Pre-dispatcher legacy patch. Zero-fills block_table tail past valid sequences so out-of-bounds prefetch doesn't read stale page indices.",
    },
    "P15": {
        "wiring": "patch_15_qwen3_none_null",
        "title": "Qwen3 None/null tool arg parser",
        "env_flag": "GENESIS_LEGACY_P15",
        "default_on": True,
        "lifecycle": "legacy",
        "category": "structured_output",
        "credit": "Pre-dispatcher legacy patch. Tolerates None / null tool arguments in Qwen3 parser instead of raising.",
    },
    "P17": {
        "title": "Marlin MoE per-SM tuning (P17/P18)",
        "env_flag": "GENESIS_LEGACY_P17",
        "default_on": True,
        "lifecycle": "legacy",
        "category": "kernel_perf",
        "credit": "Pre-dispatcher legacy patch. Per-SM (SM86) tuned configs for Marlin MoE kernel — bsm=8 selected on Ampere consumer cards.",
    },
    "P18b": {
        "title": "TurboQuant decode stage1 tune",
        "env_flag": "GENESIS_LEGACY_P18B",
        "default_on": True,
        "lifecycle": "legacy",
        "category": "kernel_perf",
        "credit": "Pre-dispatcher legacy patch. Tuned launch config for TQ decode stage1 kernel on SM86.",
    },
    "P20": {
        "title": "TurboQuant continuation-prefill FP16 rotate",
        "env_flag": "GENESIS_LEGACY_P20",
        "default_on": True,
        "lifecycle": "legacy",
        "category": "kernel_perf",
        "credit": "Pre-dispatcher legacy patch. FP16 rotation for TQ continuation-prefill path (JartX/vllm#11 prerequisite for v7.0+).",
    },
    "P22": {
        "title": "TurboQuant shared dequant prealloc",
        "env_flag": "GENESIS_LEGACY_P22",
        "default_on": True,
        "lifecycle": "legacy",
        "category": "memory_pool",
        "credit": "Pre-dispatcher legacy patch. Preallocates shared dequant scratch buffer so TQ doesn't allocate-per-step (Genesis-original).",
    },
    "P23": {
        "title": "Marlin FP32_REDUCE env override",
        "env_flag": "GENESIS_LEGACY_P23",
        "default_on": True,
        "lifecycle": "legacy",
        "category": "kernel_perf",
        "credit": "Pre-dispatcher legacy patch. Honors VLLM_MARLIN_FP32_REDUCE env to force FP32 reduction in Marlin matmul (numerical-stability hedge).",
    },
    "P24": {
        "wiring": "patch_24_moe_tune",
        "title": "fused_moe num_warps/num_stages overlay",
        "env_flag": "GENESIS_LEGACY_P24",
        "default_on": True,
        "lifecycle": "legacy",
        "category": "kernel_perf",
        "credit": "Pre-dispatcher legacy patch. Overlays SM86-tuned num_warps/num_stages on fused_moe kernel selection.",
    },
    "P26": {
        "wiring": "patch_26_prefill_output",
        "title": "TurboQuant prefill output prealloc",
        "env_flag": "GENESIS_LEGACY_P26",
        "default_on": True,
        "lifecycle": "legacy",
        "category": "memory_pool",
        "credit": "Pre-dispatcher legacy patch. Preallocates TQ prefill output buffer to avoid per-step allocation churn.",
    },
    "P27": {
        "wiring": "patch_27_reasoning_before_think",
        "title": "Qwen3 BEFORE-THINK fallback",
        "env_flag": "GENESIS_LEGACY_P27",
        "default_on": True,
        "lifecycle": "legacy",
        "category": "structured_output",
        "credit": "Pre-dispatcher legacy patch. Falls back to BEFORE-THINK parsing path when Qwen3 model emits tool_call before <think>.",
    },
    "P28": {
        "title": "GDN core_attn_out prealloc",
        "env_flag": "GENESIS_LEGACY_P28",
        "default_on": True,
        "lifecycle": "legacy",
        "category": "memory_pool",
        "credit": "Pre-dispatcher legacy patch. Preallocates GDN core_attn_out as a layer-persistent buffer + zero()-on-reuse instead of torch.zeros() per-step. Reduces allocator pressure on GDN forward.",
        "conflicts_with": ["PN32"],
    },
    "P29": {
        "title": "tool parser IndexError guard",
        "env_flag": "GENESIS_LEGACY_P29",
        "default_on": True,
        "lifecycle": "legacy",
        "category": "structured_output",
        "credit": "Pre-dispatcher legacy patch. Wraps tool-arg index access so malformed parser state returns empty instead of raising IndexError.",
    },
    "P31": {
        "title": "MoE router fp32 softmax",
        "env_flag": "GENESIS_LEGACY_P31",
        "default_on": True,
        "lifecycle": "legacy",
        "category": "model_correctness",
        "credit": "Pre-dispatcher legacy patch. Upcasts MoE router softmax to fp32 (DeepSeek-V3 pattern, deepseek_v2.py:345 reference). Improves expert routing stability on consumer Ampere.",
    },
    "P32": {
        "title": "TurboQuant cu_2 + synth_seq_lens preallocs (P32/P33)",
        "env_flag": "GENESIS_LEGACY_P32",
        "default_on": True,
        "lifecycle": "legacy",
        "category": "memory_pool",
        "credit": "Pre-dispatcher legacy patch. Preallocates cu_2 and synth_seq_lens TQ scratch tensors as persistent buffers.",
    },
    "P34": {
        "wiring": "patch_34_mamba_deadlock_guard",
        "title": "Mamba zero-collapse deadlock guard",
        "env_flag": "GENESIS_LEGACY_P34",
        "default_on": True,
        "lifecycle": "legacy",
        "category": "stability",
        "credit": "Pre-dispatcher legacy patch. Guards against Mamba state collapse-to-zero deadlock when delta is exactly zero on hybrid models.",
    },
    "P36": {
        "wiring": "patch_36_tq_shared_decode_buffers",
        "title": "TurboQuant shared decode buffers",
        "env_flag": "GENESIS_LEGACY_P36",
        "default_on": True,
        "lifecycle": "legacy",
        "category": "memory_pool",
        "credit": "Pre-dispatcher legacy patch. Shared decode-stage scratch buffers across TQ layers to amortize allocation.",
    },
    "P37": {
        "wiring": "patch_37_moe_intermediate_cache",
        "title": "MoE intermediate cache pool (opt-in)",
        # Audit P1 fix 2026-05-05 (genesis_local_consistency_audit + runtime audit):
        # registry was `GENESIS_ENABLE_P37_MOE_INTER_CACHE` but wiring code,
        # apply_all docstring, AND launch scripts all use `GENESIS_ENABLE_P37`.
        # env_flag_guard was reporting it as suspicious typo. Aligned to short form.
        "env_flag": "GENESIS_ENABLE_P37",
        "default_on": False,
        "lifecycle": "legacy",
        "category": "memory_pool",
        "credit": "Pre-dispatcher legacy patch. Opt-in pool for MoE intermediate activations. noonghunna's club-3090 long-text recipe ships with this enabled.",
    },
    "P38": {
        "title": "TQ _continuation_prefill persistent workspace",
        "env_flag": "GENESIS_LEGACY_P38",
        "default_on": True,
        "lifecycle": "legacy",
        "category": "memory_pool",
        "credit": "Pre-dispatcher legacy patch. Persistent workspace tensor for TQ continuation-prefill, addresses VolandBerlioz's OOM site at turboquant_attn.py. Companion: P38B (compile-safe in-source hook, see PATCH_REGISTRY).",
    },
    "P39a": {
        "title": "FLA chunk_scaled_dot_kkt persistent A pool",
        "env_flag": "GENESIS_LEGACY_P39A",
        "default_on": True,
        "lifecycle": "legacy",
        "category": "memory_pool",
        "credit": "Pre-dispatcher legacy patch. Persistent pool for FLA chunk_scaled_dot_kkt's A matrix to avoid per-step allocation in GDN backward.",
    },
    "P40": {
        "title": "TurboQuant GQA-grouped decode stage1 (opt-in)",
        # Audit P1 fix 2026-05-05: same class as P37 — registry was
        # `GENESIS_ENABLE_P40_GQA_GROUPED_DECODE` but wiring/kernel/scripts
        # use `GENESIS_ENABLE_P40`. compat/presets.py also used yet another
        # variant (`GENESIS_ENABLE_P40_TQ_GROUPED_DECODE`). Aligned to short form.
        "env_flag": "GENESIS_ENABLE_P40",
        "default_on": False,
        "lifecycle": "legacy",
        "category": "kernel_perf",
        "credit": "Pre-dispatcher legacy patch (vllm#40792 backport candidate). Opt-in GQA-grouped TQ decode stage1 kernel. Welch t-test on 2x A5000 single-stream: not significant (p=0.284 vs baseline 183 TPS) — kept opt-in pending Blackwell retest.",
    },
    "P44": {
        "title": "TQ mixed-batch attn_out pool",
        "env_flag": "GENESIS_LEGACY_P44",
        "default_on": True,
        "lifecycle": "legacy",
        "category": "memory_pool",
        "credit": "Pre-dispatcher legacy patch. Pool for TQ attn_out tensor under mixed prefill+decode batches.",
    },
    "P46": {
        "title": "GDN gating buffer pool",
        "env_flag": "GENESIS_LEGACY_P46",
        "default_on": True,
        "lifecycle": "legacy",
        "category": "memory_pool",
        "credit": "Pre-dispatcher legacy patch. Pool for GDN gating tensor to avoid per-layer allocation.",
    },
    "P51": {
        "title": "TQ-active runtime layer-level guard",
        "env_flag": "GENESIS_LEGACY_P51",
        "default_on": True,
        "lifecycle": "legacy",
        "category": "quantization",
        "credit": "Pre-dispatcher library patch. Runtime layer-level TQ-active detection in kernels/dequant_buffer.py — skips TQ preallocs on layers where TQ is not active. No env toggle (defensive runtime check). Companion to model_detect's config-level TQ check.",
    },
    "P102": {
        "title": "Unified spec-decode metadata + disagreement tracker (TRT-LLM style)",
        "env_flag": "GENESIS_ENABLE_P102",
        "default_on": False,
        "category": "spec_decode",
        "credit": "Genesis-original (Sander 2026-04-29). First-class spec_meta module that wraps spec-decode metadata into a unified object + tracks predicate disagreement (e.g. should_dispatch_p67 disagreements between proposer and verify paths). Diagnostic-only opt-in observability layer; emits log lines when divergence detected. Future hook for unified spec-decode dispatcher refactor.",
        "upstream_pr": None,
    },
    "PN60": {
        "title": "Quant arg vs config.json validator (preflight DX)",
        "env_flag": "GENESIS_ENABLE_PN60",
        "default_on": True,
        "category": "stability",
        "credit": "Genesis-original 2026-05-05 (apnar club-3090#51 finding). Cross-checks operator's --quantization CLI arg against the model's config.json:quantization_config.quant_method BEFORE vLLM loads. Emits one-line remediation hint instead of a 30-line pydantic ValidationError. Doctor extension; runs at preflight, no monkey-patch.",
        "upstream_pr": None,
        "applies_to": {},
    },
    "PN61": {
        "wiring": "patch_N61_qwen3_vl_keyerror_guard",
        "title": "qwen3_vl loader KeyError → text-only auto-fallback (vllm-loader guard)",
        "env_flag": "GENESIS_ENABLE_PN61",
        "default_on": False,
        "category": "stability",
        "credit": "Genesis-original 2026-05-05 (apnar club-3090#51 NVFP4 finding). Catches `KeyError: 'blocks.0.attn.proj.weight'` in qwen3_vl.load_weights when an NVFP4 quant strips the ViT tower; emits WARN + auto-sets language_model_only=True instead of crashing. Same defensive pattern as P29 IndexError guard.",
        "upstream_pr": None,
        "applies_to": {"model_class": ["qwen3_vl"]},
    },
    "PN62": {
        "wiring": "patch_N62_text_only_vit_skip",
        "title": "Text-only ViT scratch MARKER-ONLY (predicted 3-5 GiB; real hook pending)",
        "env_flag": "GENESIS_ENABLE_PN62",
        "default_on": False,
        "category": "memory_savings",
        "credit": "Genesis-original 2026-05-05 (apnar club-3090#51 highest-impact gap). When mm_limits_all_zero AND --language-model-only, the qwen3_vl visual-tower scratch allocation in gpu_model_runner._dummy_run still fires and reserves ~3-5 GiB. PN62 v1 wraps _dummy_run and SETS marker `_pn62_skip_vit_scratch=True` — but no production hook reads it yet (audit G-POST-04 honesty). Real ViT-alloc skip lands when the inner alloc helper learns to honour the marker. Predicted 3-5 GiB save on qwen3_vl + NVFP4 single-card boot pending real hook + cross-rig validation. Sister to PR35975 (text-only inputs_embeds skip, vllm#35975 merged).",
        "upstream_pr": None,
        "applies_to": {"model_class": ["qwen3_vl"]},
    },
    "PN63": {
        "title": "fp8_e5m2 advisory for consumer Blackwell (gpu_profile recommendation)",
        "env_flag": "GENESIS_ENABLE_PN63",
        "default_on": True,
        "category": "stability",
        "credit": "Genesis-original 2026-05-05 (apnar club-3090#51 empirical). Adds a per-GPU advisory entry to gpu_profile.PATCH_RECOMMENDATIONS that recommends --kv-cache-dtype fp8_e5m2 over fp8_e4m3 on consumer Blackwell (sm 12.0) until vLLM e4m3 codepath matures. Suggest-only; operator passes via CLI.",
        "upstream_pr": None,
    },
    "PN64": {
        "title": "Marlin MoE per-SM tuning placeholder for SM 12.0 (consumer Blackwell)",
        "env_flag": "GENESIS_ENABLE_PN64",
        "default_on": False,
        "category": "kernel_perf",
        "credit": "Genesis-original 2026-05-05 (apnar club-3090#51 — boot log shows `[Genesis] skipped: P17/P18 Marlin MoE per-SM tuning — no tuning entry for SM (12, 0)`). PN64 adds a placeholder entry copying SM (9, 0) Hopper config until empirical sweep data lands from sm_120. Author-blocked: needs real 5090 sweep — solicit from apnar/jhsmith409.",
        "upstream_pr": None,
        "applies_to": {},
    },
    "PN65": {
        "wiring": "patch_N65_access_log",
        "title": "Genesis structured API access log middleware (operator UX)",
        "env_flag": "GENESIS_ENABLE_PN65",
        "default_on": False,
        "category": "request_middleware",
        "credit": "Genesis-original 2026-05-05 (Sander request 'по апи лог невзрачный надо тоже проработать'). Replaces uvicorn's bare `INFO: 192.168.1.10:45116 - GET /v1/models 401 Unauthorized` with `[Genesis-API] 200  POST /v1/chat/completions  34ms  prompt=46t  completion=400t  tools=1  client=192.168.1.10`. Suppresses /health polling by default (GENESIS_PN65_LOG_HEALTH=1 to include). Status-aware level (2xx INFO / 4xx WARN / 5xx ERROR + exception type).",
        "upstream_pr": None,
        "applies_to": {},
    },
    "PR41674": {
        "wiring": "patch_pr41674_thinking_budget_inverted_bool",
        "title": "thinking_token_budget inverted-bool fix (vllm#41674 backport, 1-line)",
        "env_flag": "GENESIS_ENABLE_PR41674_THINKING_TOKEN_BUDGET_BOOL_FIX",
        "env_flag_aliases": ["GENESIS_ENABLE_PN67"],
        "default_on": False,
        "category": "stability",
        "credit": "Backport of vllm#41674 (JasonKeyiL, OPEN as of 2026-05-04). Single-token fix in `vllm/v1/worker/gpu_input_batch.py:879` — removes `not` from `or not thinking_budget_tracks_reqs`. Bug: thinking_token_budget was silently ignored for any request without penalty parameters. NULL on Genesis PROD (we don't enable thinking_token_budget); defensive for users who experiment with it. Trivial backport, zero risk.",
        "upstream_pr": 41674,
        "applies_to": {},
    },
    "PN70": {
        "wiring": "patch_N70_tool_schema_subset_filter",
        "title": "Tool schema subset filter (combined `anyOf` xgrammar-clean) — companion to P68 v7.72.1",
        "env_flag": "GENESIS_ENABLE_PN70_TOOL_SCHEMA_FILTER",
        "default_on": False,
        "category": "structured_output",
        "credit": "Genesis-original — implements lexhoefsloot's option-3 fix for noonghunna/club-3090#57. Wraps `vllm.tool_parsers.utils._get_json_schema_from_tools` and filters tools containing xgrammar-unsupported JSON Schema keys (patternProperties / propertyNames / $ref / oneOf / etc.) BEFORE the combined `anyOf` is built and handed to xgrammar. Companion to P68's option-1 skip: where P68 refuses to upgrade tool_choice on dirty catalogs, PN70 keeps the upgrade and filters dirty tools out of grammar enforcement (model can still SEE all tools in context but grammar restricts callable subset). Reuses P68's `_scan_schema_for_unsupported_key` so the unsupported-key set is single-sourced. Off by default; enable per workload.",
        "applies_to": {},
        "composes_with": ["P68"],
    },
}


# ─── Layer 2: model-aware applies_to gate ─────────────────────────────────

def _check_applies_to(
    patch_id: str, meta: dict[str, Any]
) -> tuple[bool, str]:
    """Layer 2 model-compatibility gate.

    Reads `meta["applies_to"]` (a dict mapping profile-key → list of allowed
    values), looks up the live model profile via `model_detect.get_model_profile()`,
    and returns (compatible, reason).

    Profile keys recognized: 'model_class', 'quant_format', 'kv_cache_dtype',
    'is_moe', 'is_hybrid', 'is_turboquant'. Any key whose actual value is None
    (detector couldn't resolve) is treated as compatible (conservative — let
    the patch apply and have its own call-site guards decide).

    Returns:
        (True, reason)  — model matches all applies_to constraints (or none
                          declared, or model couldn't be resolved)
        (False, reason) — explicit incompatibility: actual_value not in
                          allowed set for at least one key
    """
    applies_to = meta.get("applies_to")
    if not applies_to:
        return True, "no applies_to declared (model-class agnostic)"

    try:
        from vllm._genesis.model_detect import get_model_profile
        profile = get_model_profile()
    except Exception as e:
        return True, f"model_detect probe failed ({e}) — conservative apply"

    if not profile.get("resolved", False):
        return True, "model profile unresolved — conservative apply"

    # Map profile keys onto applies_to keys (some applies_to use boolean
    # aliases like is_moe / is_hybrid / is_turboquant for readability).
    key_aliases = {
        "is_moe": "moe",
        "is_hybrid": "hybrid",
        "is_turboquant": "turboquant",
    }

    # Build a flat profile dict that combines model_detect output with
    # boolean-alias mapping so the new predicates evaluator can read both
    # "is_turboquant" (used in applies_to) and "turboquant" (model_detect
    # native key) interchangeably.
    flat_profile: dict[str, Any] = dict(profile)
    for applies_key, profile_key in key_aliases.items():
        if profile_key in profile and applies_key not in flat_profile:
            flat_profile[applies_key] = profile[profile_key]

    # ─── Path A: richer predicate DSL (compat/predicates) ────────────────
    # Detect compound keys: all_of / any_of / not / none_of. If present,
    # delegate to the new evaluator. This lets new patches use the richer
    # syntax while old ones keep working unchanged.
    compound_keys = ("all_of", "any_of", "not", "none_of")
    if any(k in applies_to for k in compound_keys):
        try:
            from vllm._genesis.compat.predicates import evaluate
            ok, reason = evaluate(applies_to, flat_profile)
            return ok, ("applies_to satisfied" if ok
                        else f"MODEL-COMPAT: {reason}")
        except Exception as e:
            log.warning(
                "[Genesis dispatcher] %s: predicate evaluator raised (%s) — "
                "conservative apply. Check applies_to syntax.",
                patch_id, e,
            )
            return True, f"predicate evaluator error ({e}) — conservative apply"

    # ─── Path B: legacy flat-dict applies_to (backward compatible) ───────
    # Also pull version-related keys out and check via version_check.
    version_keys = (
        "vllm_version_range", "torch_version_min", "triton_version_min",
        "cuda_runtime_min", "nvidia_driver_min", "python_version_min",
        "compute_capability_min", "compute_capability_max",
    )
    version_constraints = {k: v for k, v in applies_to.items() if k in version_keys}
    profile_constraints = {k: v for k, v in applies_to.items() if k not in version_keys}

    # Version range checks
    if version_constraints:
        try:
            from vllm._genesis.compat.version_check import (
                check_version_constraints,
            )
            v_ok, v_results = check_version_constraints(version_constraints)
            if not v_ok:
                failed = [r for r in v_results if r.matched is False]
                if failed:
                    return False, f"VERSION: {failed[0].reason}"
                return False, "VERSION: constraint violation"
        except Exception as e:
            log.debug("[Genesis dispatcher] %s: version_check failed (%s) — "
                      "conservative apply", patch_id, e)

    # Legacy profile gates
    for key, allowed in profile_constraints.items():
        profile_key = key_aliases.get(key, key)
        actual = profile.get(profile_key)
        if actual is None:
            continue  # detector couldn't resolve → conservative
        if not isinstance(allowed, (list, tuple, set)):
            allowed = [allowed]
        if actual not in allowed:
            return False, (
                f"MODEL-COMPAT: {key}={actual!r} not in {list(allowed)!r}"
            )
    return True, "applies_to satisfied"


# ─── Single-call gate ─────────────────────────────────────────────────────

def should_apply(patch_id: str) -> tuple[bool, str]:
    """Unified gate: returns (apply_decision, reason).

    Combines:
      - env-flag check (`GENESIS_ENABLE_P<patch>=1` opt-in)
      - `applies_to` model-compatibility hard-skip (Layer 2, opt-in patches
        respect env override; default_on patches honor it strictly)
      - `config_detect.recommend(patch_id)` (model+config-aware decision)

    The decision rule:

      1. If env flag is truthy AND patch is opt-in (default_on=False) → apply,
         operator override wins over applies_to (logged as override).
      2. If env flag is unset/falsy AND patch is `default_on=False` → skip (opt-in)
      3. If applies_to declared and actual model profile mismatches → hard-skip
         with WARNING-class reason ("MODEL-COMPAT: ..."). For default_on=True
         patches this kicks in unconditionally; for env-truthy opt-ins it's
         logged but apply proceeds (override).
      4. Otherwise consult `config_detect.recommend()`:
         - "skip:..." → don't apply
         - "redundant:..." → don't apply
         - "deprecated:..." → don't apply
         - "neutral" / "apply" → apply

    Returns:
        (True, reason) — patch should apply
        (False, reason) — patch should skip, with human-readable reason
    """
    meta = PATCH_REGISTRY.get(patch_id)
    if meta is None:
        return False, f"unknown patch_id {patch_id!r}"

    env_flag = meta.get("env_flag")
    env_value = os.environ.get(env_flag, "") if env_flag else ""
    env_truthy = env_value.strip().lower() in ("1", "true", "yes", "on")

    # Back-compat: honor deprecated env-flag aliases (the pre-2026-06 PN##/P##
    # names superseded by the GENESIS_ENABLE_PR##### scheme). Set the new flag
    # going forward; old names keep working with a one-time deprecation log.
    if not env_truthy:
        for _alias in meta.get("env_flag_aliases", ()):
            if os.environ.get(_alias, "").strip().lower() in ("1", "true", "yes", "on"):
                env_truthy = True
                log.warning(
                    "[Genesis dispatcher] %s: env flag %s is DEPRECATED — "
                    "use %s instead (still honored for now).",
                    patch_id, _alias, env_flag,
                )
                break

    # Operator override: env truthy = always apply (subject to anchor presence)
    if env_truthy:
        # Layer 2 applies_to is informational under env-override
        compat, compat_reason = _check_applies_to(patch_id, meta)
        if not compat:
            log.warning(
                "[Genesis dispatcher] %s: env OVERRIDE applies_to mismatch — "
                "%s. Proceeding because operator set %s=1.",
                patch_id, compat_reason, env_flag,
            )
        # Still consult config_detect to PRINT the recommendation as info
        try:
            from vllm._genesis.config_detect import recommend
            verdict, reason = recommend(patch_id)
            if verdict == "apply":
                return True, f"opt-in env + config recommends apply: {reason}"
            elif verdict == "neutral":
                return True, f"opt-in env (config: neutral)"
            else:
                return True, (
                    f"opt-in env OVERRIDE (config recommends {verdict}: "
                    f"{reason}) — proceeding because operator forced it"
                )
        except Exception as e:
            return True, f"opt-in env (config_detect probe failed: {e})"

    # Env flag unset/falsy
    if not meta.get("default_on", False):
        if meta.get("deprecated", False):
            return False, (
                f"opt-in only AND empirically deprecated — "
                f"keeping skip; set {env_flag}=1 only for diagnostics"
            )
        return False, f"opt-in only — set {env_flag}=1 to engage"

    # default_on=True: enforce applies_to as Layer 2 HARD skip.
    compat, compat_reason = _check_applies_to(patch_id, meta)
    if not compat:
        log.warning(
            "[Genesis dispatcher] %s HARD-SKIP — %s. Patch designed for a "
            "different model class; skipping to avoid overhead. Set %s=1 to "
            "force-apply if you know what you are doing.",
            patch_id, compat_reason, env_flag,
        )
        return False, compat_reason

    # default_on=True patches still consult config_detect
    try:
        from vllm._genesis.config_detect import recommend
        verdict, reason = recommend(patch_id)
        return (verdict in ("apply", "neutral")), f"config_detect: {verdict}:{reason}"
    except Exception as e:
        return False, f"config_detect failed: {e}"


# ─── Decision logging ─────────────────────────────────────────────────────

# Module-level cache of decisions made this boot, for matrix dump.
_DECISIONS: list[dict[str, Any]] = []


def log_decision(patch_id: str, applied: bool, reason: str) -> None:
    """Log + record a patch decision for the boot-time matrix dump.

    Single condensed line per patch. Operator can see all decisions at boot
    via `Genesis Dispatcher v2 decisions:` log block (called from apply_all).
    """
    meta = PATCH_REGISTRY.get(patch_id, {})
    title = meta.get("title", patch_id)
    status = "APPLY" if applied else "SKIP "
    log.info(
        "[Genesis Dispatcher] %s %s — %s | %s",
        status, patch_id, title, reason[:120],
    )
    _DECISIONS.append({
        "patch_id": patch_id,
        "title": title,
        "applied": applied,
        "reason": reason,
        "env_flag": meta.get("env_flag", ""),
        "credit": meta.get("credit", ""),
        "upstream_pr": meta.get("upstream_pr"),
    })


def get_apply_matrix() -> list[dict[str, Any]]:
    """Return the recorded apply matrix for this boot.

    Useful for tests + diagnostic dump.
    """
    return list(_DECISIONS)


def dump_apply_matrix() -> str:
    """Format the apply matrix as ASCII table (string for printing).

    Columns: patch_id, status, title, reason (truncated), credit.
    """
    if not _DECISIONS:
        return "(no decisions recorded — Genesis Dispatcher hasn't been used yet)"

    # Compute column widths
    rows = [
        (
            d["patch_id"],
            "APPLY" if d["applied"] else "SKIP",
            d["title"][:45],
            d["reason"][:60],
            d.get("credit", "")[:30],
        )
        for d in _DECISIONS
    ]
    widths = [max(len(r[i]) for r in rows) for i in range(5)]
    widths = [max(w, len(h)) for w, h in zip(widths,
              ["Patch", "Status", "Title", "Reason", "Credit"])]

    def _fmt_row(r):
        return " | ".join(c.ljust(widths[i]) for i, c in enumerate(r))

    lines = []
    lines.append(_fmt_row(["Patch", "Status", "Title", "Reason", "Credit"]))
    lines.append("-+-".join("-" * w for w in widths))
    for r in rows:
        lines.append(_fmt_row(r))
    return "\n".join(lines)


def log_apply_matrix() -> None:
    """Emit the apply matrix as a multi-line INFO block.

    Called by apply_all at end of boot to give operator a single readable
    summary instead of grep-ing through scattered INFO lines.
    """
    matrix = dump_apply_matrix()
    log.info(
        "[Genesis Dispatcher v2] apply matrix:\n%s",
        matrix,
    )


def dump_structured_boot_summary() -> str:
    """Emit a structured, table-formatted boot summary.

    Sections:
      1. System info — GPU, vllm pin, Genesis version, model class
      2. Per-category APPLY/SKIP/FAIL counters
      3. APPLIED patches table (grouped by category)
      4. SKIPPED patches (grouped by reason class: env-disabled / model-incompat
         / upstream-merged / conflict / other)
      5. FAILED patches (highlighted, none expected in healthy boot)
      6. Active warnings (regression-flagged enabled patches)

    Designed for readability by operators who tail container logs. Replaces
    the scattered per-patch INFO lines + the bare apply matrix.
    """
    if not _DECISIONS:
        return "(no Genesis decisions recorded — patcher not active or first call)"

    # Dedup: keep last decision per patch_id (handles multi-worker boot
    # where apply_all runs once per TP rank — the second call typically
    # logs `already applied (idempotent)` for the same patches).
    _seen: dict[str, dict[str, Any]] = {}
    for d in _DECISIONS:
        _seen[d["patch_id"]] = d
    decisions = list(_seen.values())

    lines: list[str] = []

    # ─── 1. System info header ────────────────────────────────────────────
    lines.append("═" * 78)
    lines.append("Genesis vLLM Patcher — boot summary")
    lines.append("═" * 78)

    # Genesis version
    try:
        from vllm._genesis import __version__ as _gver
        _gver_str = _gver.lstrip("v")  # avoid "vv7.63.x" if module already prefixes
        lines.append(f"  Genesis:  v{_gver_str}")
    except Exception:
        lines.append("  Genesis:  (version unavailable)")

    # vllm pin
    try:
        import vllm as _vllm
        lines.append(f"  vLLM:     {getattr(_vllm, '__version__', 'unknown')}")
    except Exception:
        lines.append("  vLLM:     (import failed)")

    # GPU + compute capability
    try:
        import torch
        if torch.cuda.is_available():
            n = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            cap = torch.cuda.get_device_capability(0)
            lines.append(
                f"  GPU:      {n}× {gpu_name} (sm_{cap[0]}{cap[1]})"
            )
    except Exception:
        pass

    # Model profile (if loaded)
    try:
        from vllm._genesis.model_detect import get_model_profile
        profile = get_model_profile()
        if profile.get("resolved", False):
            mc = profile.get("model_class", "unknown")
            qf = profile.get("quant_format", "unknown")
            kv = profile.get("kv_cache_dtype", "unknown")
            hyb = "hybrid" if profile.get("hybrid") else "dense"
            lines.append(
                f"  Model:    {mc} | quant={qf} | kv={kv} | {hyb}"
            )
    except Exception:
        pass

    # ─── 2. Counters ──────────────────────────────────────────────────────
    n_apply = sum(1 for d in decisions if d["applied"])
    n_skip = sum(1 for d in decisions if not d["applied"])
    lines.append("─" * 78)
    lines.append(
        f"  Patches:  {len(decisions)} total  →  "
        f"{n_apply} APPLY  |  {n_skip} SKIP"
    )

    # Per-category breakdown
    cat_counts: dict[str, dict[str, int]] = {}
    for d in decisions:
        meta = PATCH_REGISTRY.get(d["patch_id"], {})
        cat = meta.get("category", "uncategorized")
        bucket = cat_counts.setdefault(cat, {"apply": 0, "skip": 0})
        bucket["apply" if d["applied"] else "skip"] += 1

    if cat_counts:
        lines.append("  By category:")
        for cat in sorted(cat_counts):
            c = cat_counts[cat]
            lines.append(
                f"    • {cat:<22} APPLY={c['apply']:>3}  SKIP={c['skip']:>3}"
            )

    # ─── Pretty category labels ──────────────────────────────────────────
    # Friendly human-readable description for each registry category.
    CATEGORY_LABELS = {
        "compile_safety":      "Compile / cudagraph safety",
        "hybrid":              "Hybrid GDN / Mamba (qwen3_5/3_6)",
        "kernel":              "Kernel correctness (Marlin / TQ)",
        "kernel_perf":         "Kernel performance tuning",
        "kernel_safety":       "Kernel-level safety guards",
        "kv_cache":            "KV cache management",
        "memory_hotfix":       "Memory hotfix (Cliff 2 / OOM)",
        "memory_pool":         "Memory pool / scratch buffers",
        "memory_savings":      "Memory savings (defensive)",
        "model_correctness":   "Model correctness (load / dtype)",
        "perf_hotfix":         "Performance hotfix (defensive)",
        "perf_kernel":         "Performance kernel rewrite",
        "quantization":        "Quantization (AutoRound / FP8)",
        "request_middleware":  "Request middleware",
        "spec_decode":         "Speculative decoding (MTP / ngram)",
        "stability":           "Stability / DX safeguards",
        "structured_output":   "Structured output / Qwen3 parser",
        "uncategorized":       "Uncategorized",
    }

    def _cat_label(cat: str) -> str:
        return CATEGORY_LABELS.get(cat, cat)

    # ─── 3. APPLIED patches grouped by category ──────────────────────────
    applied_by_cat: dict[str, list[dict[str, Any]]] = {}
    for d in decisions:
        if not d["applied"]:
            continue
        meta = PATCH_REGISTRY.get(d["patch_id"], {})
        cat = meta.get("category", "uncategorized")
        applied_by_cat.setdefault(cat, []).append(d)

    if applied_by_cat:
        lines.append("─" * 78)
        lines.append(f"  ✓ APPLIED ({n_apply})")
        for cat in sorted(applied_by_cat):
            label = _cat_label(cat)
            count = len(applied_by_cat[cat])
            lines.append("")
            lines.append(f"  ╔═══ {label} ({count})")
            for d in applied_by_cat[cat]:
                upstream = ""
                meta = PATCH_REGISTRY.get(d["patch_id"], {})
                if meta.get("upstream_pr"):
                    upstream = f"  ←  vllm#{meta['upstream_pr']}"
                lines.append(
                    f"  ║   • {d['patch_id']:<10}  {d['title'][:90]}{upstream}"
                )

    # ─── 4. SKIPPED patches grouped by reason class ──────────────────────
    skip_classes = {
        "upstream_merged": [],
        "env_disabled": [],
        "model_incompat": [],
        "conflict": [],
        "other": [],
    }
    for d in decisions:
        if d["applied"]:
            continue
        reason = d["reason"].lower()
        if "upstream" in reason and ("merged" in reason or "drift" in reason):
            cls = "upstream_merged"
        elif "opt-in" in reason or "set genesis_enable" in reason:
            cls = "env_disabled"
        elif "applies_to" in reason or "incompatible" in reason or \
                "model_class" in reason or "no gdn" in reason:
            cls = "model_incompat"
        elif "conflict" in reason or "mutually exclusive" in reason or \
                "skipped — p" in reason:
            cls = "conflict"
        else:
            cls = "other"
        skip_classes[cls].append(d)

    SKIP_LABELS = {
        "upstream_merged": "Upstream merged in current pin (auto-skip)",
        "env_disabled":    "Opt-in (env flag disabled by operator)",
        "model_incompat":  "Model architecture incompatible (applies_to)",
        "conflict":        "Conflict / mutual-exclusion with active patch",
        "other":           "Other / config-neutral",
    }

    if n_skip > 0:
        lines.append("")
        lines.append("─" * 78)
        lines.append(f"  ⊘ SKIPPED ({n_skip}) — grouped by reason")
        for cls, items in skip_classes.items():
            if not items:
                continue
            label = SKIP_LABELS.get(cls, cls)
            lines.append("")
            lines.append(f"  ╔═══ {label} ({len(items)})")
            for d in items[:12]:  # cap per-class to keep summary readable
                lines.append(
                    f"  ║   • {d['patch_id']:<10}  {d['title'][:90]}"
                )
            if len(items) > 12:
                lines.append(f"  ║   … and {len(items) - 12} more")

    # ─── 5. FAILED (highlighted) ─────────────────────────────────────────
    failed = [d for d in decisions
              if not d["applied"] and "fail" in d["reason"].lower()]
    if failed:
        lines.append("─" * 78)
        lines.append(f"  ⚠ FAILED ({len(failed)}) — investigate before serving traffic")
        for d in failed:
            lines.append(
                f"    {d['patch_id']:<8}  {d['title'][:50]}"
            )
            lines.append(f"             reason: {d['reason'][:65]}")

    lines.append("═" * 78)
    return "\n".join(lines)


def log_structured_boot_summary() -> None:
    """Emit the structured boot summary as a single multi-line INFO block.

    Drop-in replacement for `log_apply_matrix()`. Called once at end of
    apply_all.run() boot. Operator-friendly: tables, counters, system info,
    grouped by category and skip-reason class.
    """
    summary = dump_structured_boot_summary()
    log.info(
        "[Genesis] structured boot summary:\n%s",
        summary,
    )


# ─── A3/D2 — PATCH_REGISTRY dependency / conflict validator ───────────────
# Two layers:
#   1. validate_registry()      — static structural check (boot-time)
#   2. validate_apply_plan(set) — runtime check on actual decisions
#
# Patch metadata may declare:
#   "requires_patches": ["PR40738"]      — list of patch_ids that must also apply
#   "conflicts_with":   ["P65"]      — list of patch_ids that MUST NOT apply
# Both fields default to [] when absent (no relationship declared).


def _coerce_list(value: Any) -> list[str]:
    """Normalize a metadata field into list[str]. Tolerates None / scalar."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, set)):
        return [str(v) for v in value]
    return []


def validate_registry(
    registry: dict[str, dict[str, Any]] | None = None,
) -> list[ValidationIssue]:
    """Static validation of PATCH_REGISTRY shape.

    Checks (per declared `requires_patches` / `conflicts_with` field):
      - Every referenced id exists in the registry (else ERROR — typo class)
      - No patch references itself (else ERROR — clear bug)
      - No requires-cycle A→…→A (else ERROR — boot would loop or wedge)

    Returns a list of `ValidationIssue` (empty list = clean).
    """
    if registry is None:
        registry = PATCH_REGISTRY

    issues: list[ValidationIssue] = []
    keys = set(registry.keys())

    # 1. Reference existence + self-reference
    for pid, meta in registry.items():
        for ref in _coerce_list(meta.get("requires_patches")):
            if ref == pid:
                issues.append(ValidationIssue(
                    "ERROR", pid,
                    f"requires_patches contains self-reference {ref!r}",
                ))
            elif ref not in keys:
                issues.append(ValidationIssue(
                    "ERROR", pid,
                    f"requires_patches references unknown patch_id {ref!r}",
                ))
        for ref in _coerce_list(meta.get("conflicts_with")):
            if ref == pid:
                issues.append(ValidationIssue(
                    "ERROR", pid,
                    f"conflicts_with contains self-reference {ref!r}",
                ))
            elif ref not in keys:
                issues.append(ValidationIssue(
                    "ERROR", pid,
                    f"conflicts_with references unknown patch_id {ref!r}",
                ))

    # 2. Cycle detection on requires_patches graph (DFS three-color).
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {pid: WHITE for pid in registry}

    def _walk(pid: str, path: list[str]) -> None:
        if color[pid] == GRAY:
            cycle = path[path.index(pid):] + [pid]
            issues.append(ValidationIssue(
                "ERROR", pid,
                f"requires_patches cycle detected: {' → '.join(cycle)}",
            ))
            return
        if color[pid] == BLACK:
            return
        color[pid] = GRAY
        for ref in _coerce_list(registry.get(pid, {}).get("requires_patches")):
            if ref in color:
                _walk(ref, path + [pid])
        color[pid] = BLACK

    for pid in list(registry):
        if color[pid] == WHITE:
            _walk(pid, [])

    return issues


def validate_apply_plan(
    applied: set[str],
    registry: dict[str, dict[str, Any]] | None = None,
) -> list[ValidationIssue]:
    """Runtime validation: given the live APPLY set, surface dependency /
    conflict violations.

    Args:
        applied: set of patch_ids that the dispatcher actually decided to
            APPLY this boot (from `get_apply_matrix()` filtered by
            applied=True, or computed externally).
        registry: optional override for testing; defaults to PATCH_REGISTRY.

    Returns:
        list of ValidationIssue. Severities:
          - ERROR  : missing required, conflict-pair both applied
          - WARNING: applied set contains a patch_id not in registry

    Conflict pairs are reported once (canonicalized — sorted ids) even when
    the conflict is declared symmetrically on both sides.
    """
    if registry is None:
        registry = PATCH_REGISTRY

    issues: list[ValidationIssue] = []

    # Unknown ids in applied set
    for pid in applied:
        if pid not in registry:
            issues.append(ValidationIssue(
                "WARNING", pid,
                f"applied set contains unknown patch_id {pid!r}",
            ))

    # Required dependencies — only check patches that ARE applied
    for pid in applied:
        meta = registry.get(pid)
        if meta is None:
            continue  # already reported as unknown above
        for ref in _coerce_list(meta.get("requires_patches")):
            if ref not in applied:
                issues.append(ValidationIssue(
                    "ERROR", pid,
                    f"missing required dependency: {pid} requires {ref!r} "
                    f"to also be APPLY (currently SKIP)",
                ))

    # Conflicts — canonicalize pairs to avoid double-reporting
    seen_pairs: set[tuple[str, str]] = set()
    for pid in applied:
        meta = registry.get(pid)
        if meta is None:
            continue
        for ref in _coerce_list(meta.get("conflicts_with")):
            if ref in applied:
                pair = tuple(sorted([pid, ref]))
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                issues.append(ValidationIssue(
                    "ERROR", pid,
                    f"conflict: {pair[0]} and {pair[1]} are both APPLY but "
                    f"declared mutually exclusive — pick one",
                ))

    return issues


def log_validation_issues(issues: list[ValidationIssue]) -> None:
    """Emit issues at appropriate log severity. Operator-readable summary."""
    if not issues:
        log.info("[Genesis Dispatcher v2] validator: clean (no issues)")
        return
    for i in issues:
        msg = f"[Genesis Dispatcher v2] validator {i.severity}: {i.patch_id} — {i.message}"
        if i.severity == "ERROR":
            log.error(msg)
        elif i.severity == "WARNING":
            log.warning(msg)
        else:
            log.info(msg)


# ─── CLI entry-point ──────────────────────────────────────────────────────

def main() -> int:
    """`python3 -m vllm._genesis.dispatcher` — print apply matrix as table.

    Useful for diagnostics OUTSIDE a vllm boot (e.g. dry-run profiling).
    Walks the patch registry and dry-evaluates should_apply() for each.
    """
    print("Genesis Dispatcher v2 — patch decision matrix")
    print("=" * 80)
    print()

    # Run should_apply against every registered patch
    for patch_id in PATCH_REGISTRY:
        decision, reason = should_apply(patch_id)
        log_decision(patch_id, decision, reason)

    print(dump_apply_matrix())
    print()
    print("Note: this is a STATIC dispatch view. Some recommendations are")
    print("'skip' because vllm config isn't set in this dry-run context.")
    print("In real boot, get_runtime_profile() returns the actual config.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
