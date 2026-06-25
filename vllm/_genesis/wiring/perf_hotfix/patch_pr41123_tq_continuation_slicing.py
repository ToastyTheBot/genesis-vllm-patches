# SPDX-License-Identifier: Apache-2.0
"""Wiring for Patch 101 — TQ continuation prefill 64-token slicing (vllm#41123 SELECTIVE).

Backport (SELECTIVE) of vllm#41123 "fix(kv-cache): allow TurboQuant on hybrid models".
PR open 2026-04-28.

WHAT WE TAKE (the perf-positive bits):
1. Lower _CONTINUATION_DECODE_THRESHOLD: 128 → 64 (matches A5000 SM stride; P67
   already uses 64-token tiles).
2. New constant _CONTINUATION_DECODE_MAX_CACHED_LEN = 32768 — switches to TQ
   decode kernel once cached prefix exceeds 32K (we run 320K context).
3. 64-token slicing loop in `_prefill_attention` continuation branch — bounds
   decode scratch memory, better L2 reuse on Ampere.

WHAT WE SKIP (regressions for our PROD):
- cudagraph_support downgrade (UNIFORM_BATCH → UNIFORM_SINGLE_TOKEN_DECODE)
  would force spec-decode draft batches off graph → loses ~5-8% TPS on PROD.
- Hybrid boundary-skip removal (Mechanism 1 in PR) — our launch passes explicit
  --kv-cache-dtype-skip-layers; the new explicit-skip rejection would BREAK boot.

EXPECTED IMPACT (per agent ace4cf3dcd2969e15 analysis):
- Typical 8-32K continuation: +3-6% TPS (less dequant scratch, better L2)
- Long cached prefix (>32K): +8-12% TPS (avoids fp16 materialization OOM-edge)
- Total projected: 198-205 TPS sustained on PROD (currently 187-193)

COMPOSABILITY:
- Composes with PR40941/PR40941b (different anchor: PR40941/PR40941b touch _decode_attention
  + workspace.py; PR41123 touches _prefill_attention continuation branch)
- P56 NEEDS RE-ANCHOR after PR41123: P56 anchors on
  `if q_len <= _CONTINUATION_DECODE_THRESHOLD:` which PR41123 replaces with
  `use_decode_continuation = (q_len <= ... or cached_len >= ...)` block
- TODO next session: update P56 to anchor on the new `use_decode_continuation`

Status: opt-in via `GENESIS_ENABLE_PR41123_TQ_CONTINUATION_64TOK_SLICE=1`. Default OFF.
applies_to: turboquant_* KV dtypes only.

Author backport: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
Original PR: vllm#41123 (selectively).
"""
from __future__ import annotations

import logging
import os

from vllm._genesis.guards import resolve_vllm_file, vllm_install_root
from vllm._genesis.wiring.text_patch import (
    TextPatch,
    TextPatcher,
    TextPatchResult,
)

log = logging.getLogger("genesis.wiring.p101_tq_continuation_slicing")


GENESIS_PR41123_MARKER = (
    "Genesis PR41123 TQ continuation 64-token slicing (vllm#41123 selective) v7.62.16"
)


# ─── Sub-patch 1: lower threshold + add MAX_CACHED_LEN constant ─────────

PR41123_THRESHOLD_OLD = (
    "# do_kv_cache_update already stored all tokens to TQ cache, so the decode\n"
    "# kernel can read them efficiently. This avoids O(cached_len) dequant work\n"
    "# per continuation, eliminating the O(N²/chunk_size) collapse at long context.\n"
    "_CONTINUATION_DECODE_THRESHOLD = 128\n"
)

PR41123_THRESHOLD_NEW = (
    "# do_kv_cache_update already stored all tokens to TQ cache, so the decode\n"
    "# kernel can read them efficiently. This avoids O(cached_len) dequant work\n"
    "# per continuation, eliminating the O(N²/chunk_size) collapse at long context.\n"
    "# [Genesis PR41123 vllm#41123 selective backport] threshold lowered 128→64\n"
    "# (matches A5000 SM stride). Added MAX_CACHED_LEN to switch back to TQ decode\n"
    "# kernel once cached prefix exceeds 32K (memory-bound on long ctx).\n"
    "_CONTINUATION_DECODE_THRESHOLD = 64\n"
    "_CONTINUATION_DECODE_MAX_CACHED_LEN = 32768\n"
)


# ─── Sub-patch 2: 64-token slicing loop in _prefill_attention ───────────
# Anchor on the old `if q_len <= _CONTINUATION_DECODE_THRESHOLD:` block.
# Replace with use_decode_continuation OR-cond + sliced loop.

PR41123_LOOP_OLD = (
    "                # Continuation chunk: tokens already stored to TQ cache\n"
    "                # by do_kv_cache_update. Use decode kernel directly to\n"
    "                # avoid O(cached_len) full-dequant per continuation.\n"
    "                # For large continuations, fall back to _continuation_prefill.\n"
    "                cached_len = seq_len - q_len\n"
    "                if q_len <= _CONTINUATION_DECODE_THRESHOLD:\n"
    "                    # Fast path: treat each query as a decode request\n"
    "                    # with incremental seq_lens for causal masking.\n"
    "                    # Slice from pre-built arange (no kernel launch)\n"
    "                    synth_seq_lens = _arange_cache[cached_len + 1 : seq_len + 1]\n"
    "                    synth_bt = attn_metadata.block_table[i : i + 1].expand(q_len, -1)\n"
    "                    out = triton_turboquant_decode_attention(\n"
    "                        query=q_seq,\n"
    "                        kv_cache=kv_cache,\n"
    "                        block_table=synth_bt,\n"
    "                        seq_lens=synth_seq_lens,\n"
    "                        Pi=Pi,\n"
    "                        centroids=centroids,\n"
    "                        scale=self.scale,\n"
    "                        mse_bits=self.tq_config.key_mse_bits,\n"
    "                        key_packed_size=self.tq_config.key_packed_size,\n"
    "                        value_quant_bits=(self.tq_config.effective_value_quant_bits),\n"
    "                        key_fp8=self.tq_config.key_fp8,\n"
    "                        norm_correction=self.tq_config.norm_correction,\n"
    "                        PiT=PiT,\n"
    "                    )\n"
)

PR41123_LOOP_NEW = (
    "                # Continuation chunk: tokens already stored to TQ cache\n"
    "                # by do_kv_cache_update. Use decode kernel directly to\n"
    "                # avoid O(cached_len) full-dequant per continuation.\n"
    "                # [Genesis PR41123 vllm#41123 selective backport]\n"
    "                # Moderate continuations still use _continuation_prefill for\n"
    "                # throughput, while long cached prefixes stay memory bounded.\n"
    "                cached_len = seq_len - q_len\n"
    "                use_decode_continuation = (\n"
    "                    q_len <= _CONTINUATION_DECODE_THRESHOLD\n"
    "                    or cached_len >= _CONTINUATION_DECODE_MAX_CACHED_LEN\n"
    "                )\n"
    "                if use_decode_continuation:\n"
    "                    # Decode path: treat each query as a decode request\n"
    "                    # with incremental seq_lens for causal masking. Keep\n"
    "                    # large chunks sliced to bound decode scratch memory.\n"
    "                    out = torch.empty_like(q_seq)\n"
    "                    for q_offset in range(0, q_len, _CONTINUATION_DECODE_THRESHOLD):\n"
    "                        q_next = min(q_offset + _CONTINUATION_DECODE_THRESHOLD, q_len)\n"
    "                        q_part = q_seq[q_offset:q_next]\n"
    "                        part_len = q_next - q_offset\n"
    "                        output_part = out[q_offset:q_next]\n"
    "                        # Slice from pre-built arange (no kernel launch)\n"
    "                        synth_seq_lens = _arange_cache[\n"
    "                            cached_len + q_offset + 1 : cached_len + q_next + 1\n"
    "                        ]\n"
    "                        synth_bt = attn_metadata.block_table[i : i + 1].expand(\n"
    "                            part_len, -1\n"
    "                        )\n"
    "                        triton_turboquant_decode_attention(\n"
    "                            query=q_part,\n"
    "                            kv_cache=kv_cache,\n"
    "                            block_table=synth_bt,\n"
    "                            seq_lens=synth_seq_lens,\n"
    "                            Pi=Pi,\n"
    "                            centroids=centroids,\n"
    "                            scale=self.scale,\n"
    "                            mse_bits=self.tq_config.key_mse_bits,\n"
    "                            key_packed_size=self.tq_config.key_packed_size,\n"
    "                            value_quant_bits=(\n"
    "                                self.tq_config.effective_value_quant_bits\n"
    "                            ),\n"
    "                            key_fp8=self.tq_config.key_fp8,\n"
    "                            norm_correction=self.tq_config.norm_correction,\n"
    "                            PiT=PiT,\n"
    "                            output_buf=output_part,\n"
    "                            buf_holder=layer,\n"
    "                            max_num_kv_splits=self.max_num_kv_splits,\n"
    "                        )\n"
)


def _make_patcher() -> TextPatcher | None:
    target = resolve_vllm_file("v1/attention/backends/turboquant_attn.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name="PR41123 turboquant_attn.py — TQ continuation 64-token slicing (vllm#41123)",
        target_file=str(target),
        marker=GENESIS_PR41123_MARKER,
        sub_patches=[
            TextPatch(
                name="p101_threshold_lower_and_max_cached",
                anchor=PR41123_THRESHOLD_OLD,
                replacement=PR41123_THRESHOLD_NEW,
                required=True,
            ),
            TextPatch(
                name="p101_continuation_slicing_loop",
                anchor=PR41123_LOOP_OLD,
                replacement=PR41123_LOOP_NEW,
                required=True,
            ),
        ],
        upstream_drift_markers=[
            "[Genesis PR41123",
            # Upstream-side markers if vllm#41123 (or equivalent) merges:
            "_CONTINUATION_DECODE_MAX_CACHED_LEN",
            "use_decode_continuation",
        ],
    )


def apply() -> tuple[str, str]:
    """Apply PR41123 — TQ continuation slicing."""
    from vllm._genesis.dispatcher import log_decision, should_apply

    decision, reason = should_apply("PR41123")
    log_decision("PR41123", decision, reason)
    if not decision:
        return "skipped", reason

    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    patcher = _make_patcher()
    if patcher is None:
        return "skipped", "turboquant_attn.py not found"

    if not os.path.isfile(patcher.target_file):
        return "skipped", f"target disappeared: {patcher.target_file}"
    with open(patcher.target_file) as f:
        content = f.read()
    if patcher.marker in content:
        log.info("[PR41123] marker present — skip (idempotent)")
        return "applied", "idempotent (marker present)"
    for m in patcher.upstream_drift_markers:
        if m.startswith("[Genesis"):
            continue
        if m in content:
            return (
                "skipped",
                f"upstream drift marker {m!r} in {patcher.target_file} "
                "— upstream PR #41123 (or equivalent) appears merged",
            )

    result, failure = patcher.apply()
    # Audit P1 fix 2026-05-05: surface SKIPPED as skipped (was masked as applied)
    if result == TextPatchResult.SKIPPED:
        _r = failure.reason if failure else "anchor drift / not eligible"
        _d = f" ({failure.detail})" if (failure and failure.detail) else ""
        return "skipped", f"{patcher.patch_name}: {_r}{_d}"
    if result == TextPatchResult.FAILED:
        return "failed", (
            f"{patcher.patch_name}: "
            f"{failure.reason if failure else 'unknown'} "
            f"({failure.detail if failure else ''})"
        )

    return (
        "applied",
        "PR41123 v7.62.16 applied: turboquant_attn.py continuation prefill now "
        "uses 64-token slicing + 32K cached-len cutoff. Expected: +3-12% TPS "
        "on PROD long-context. Composes with PR40941/PR40941b (non-overlapping anchors). "
        "REMINDER: P56 needs re-anchor against new use_decode_continuation block."
    )


def is_applied() -> bool:
    if vllm_install_root() is None:
        return False
    patcher = _make_patcher()
    if patcher is None:
        return False
    try:
        with open(patcher.target_file) as f:
            return patcher.marker in f.read()
    except Exception:
        return False
