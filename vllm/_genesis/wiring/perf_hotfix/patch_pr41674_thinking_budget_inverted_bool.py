# SPDX-License-Identifier: Apache-2.0
"""Wiring for Patch PR41674 — thinking_token_budget inverted-bool fix.

Backport of vllm-project/vllm#41674 (JasonKeyiL, OPEN as of 2026-05-04).
Trivial 1-line fix: removes `not` from the inverted boolean condition
in `vllm/v1/worker/gpu_input_batch.py:894` that caused
`thinking_token_budget` to be silently ignored for any request without
penalty parameters.

================================================================
THE BUG
================================================================

Original code (current pin, gpu_input_batch.py:890-895):

    needs_output_token_ids = (
        not self.no_penalties
        or bool(self.bad_words_token_ids)
        or self.logitsprocs_need_output_token_ids
        or not thinking_budget_tracks_reqs    # ← BUG: should not have `not`
    )

When `thinking_budget_tracks_reqs == True` AND no penalties set,
`needs_output_token_ids` was forced to True via the `not True == False`
short-circuit failure → `output_token_ids` became `[]` →
`ThinkingBudgetStateHolder.update_state()` iterated zero elements →
`_update_think_state()` never called → budget enforcement bypassed.

Fix: remove the `not` so the condition reads `or thinking_budget_tracks_reqs`.

================================================================
GENESIS APPROACH
================================================================

Single 1-line text-patch. Anchor is unique in current pin
(`0.20.2rc1.dev9+g01d4d1ad3`).

================================================================
RELATIONSHIP TO OTHER GENESIS PATCHES
================================================================

PR41674 touches `gpu_input_batch.py` — same file as PR41411 (prompt_logprobs
eviction). Different lines (PR41411 ~876 vs PR41674 ~894). Composes cleanly.

================================================================
WHO THIS HELPS
================================================================

Genesis PROD does NOT enable `thinking_token_budget`. Defensive backport
for any operator who experiments with it (e.g. testing budget caps for
agentic workflows, RAG pre-roll scoping, or 27B long-ctx context budget).

================================================================
ENV
================================================================

GENESIS_ENABLE_PR41674=1

================================================================
RISK
================================================================

ZERO — single-token removal, semantically inverts a clearly-buggy
condition. Worst case: no behavior change (we don't use the feature).

Author: Sandermage 2026-05-05.
Backport reference: vllm#41674 (JasonKeyiL, OPEN as of 2026-05-04).
"""
from __future__ import annotations

import logging
import os

from vllm._genesis.guards import resolve_vllm_file, vllm_install_root
from vllm._genesis.wiring.text_patch import (
    TextPatch,
    TextPatcher,
    result_to_wiring_status,
)

log = logging.getLogger("genesis.wiring.pn67_thinking_budget_inverted_bool")

GENESIS_PR41674_MARKER = "Genesis PR41674 thinking_token_budget inverted bool fix vllm#41674"

# Anchor: 4-line context around the buggy condition for uniqueness
PR41674_OLD = (
    "            or self.logitsprocs_need_output_token_ids\n"
    "            or not thinking_budget_tracks_reqs\n"
    "        )\n"
)
PR41674_NEW = (
    "            or self.logitsprocs_need_output_token_ids\n"
    "            # [Genesis PR41674 thinking_token_budget inverted bool fix vllm#41674]\n"
    "            # Removed `not` — was silently disabling budget for any request\n"
    "            # without penalty params (the vast majority of real-world usage).\n"
    "            or thinking_budget_tracks_reqs\n"
    "        )\n"
)


def _make_patcher() -> TextPatcher | None:
    target = resolve_vllm_file("v1/worker/gpu_input_batch.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name="PR41674 gpu_input_batch.py — thinking_token_budget inverted bool fix",
        target_file=str(target),
        marker=GENESIS_PR41674_MARKER,
        sub_patches=[
            TextPatch(
                name="pn67_remove_not",
                anchor=PR41674_OLD,
                replacement=PR41674_NEW,
                required=True,
            ),
        ],
        upstream_drift_markers=[
            "[Genesis PR41674",
        ],
    )


def apply() -> tuple[str, str]:
    """Apply PR41674 — single-token thinking_budget inverted-bool fix."""
    from vllm._genesis.dispatcher import log_decision, should_apply

    decision, reason = should_apply("PR41674")
    log_decision("PR41674", decision, reason)
    if not decision:
        return "skipped", reason

    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    patcher = _make_patcher()
    if patcher is None:
        return "skipped", "vllm/v1/worker/gpu_input_batch.py not resolvable"

    if not os.path.isfile(patcher.target_file):
        return "skipped", f"target file disappeared: {patcher.target_file}"

    # Pre-flight: detect upstream-merged auto-skip
    with open(patcher.target_file) as f:
        content = f.read()
    if "or not thinking_budget_tracks_reqs" not in content:
        return (
            "skipped",
            "abstract anchor `or not thinking_budget_tracks_reqs` no longer "
            "present — upstream PR #41674 (or equivalent) appears merged",
        )

    if patcher.marker in content:
        return "applied", "PR41674 already applied (marker present, idempotent)"

    result, failure = patcher.apply()
    return result_to_wiring_status(
        result, failure,
        applied_message=(
            "PR41674 applied: removed inverted `not` from thinking_token_budget "
            "condition in gpu_input_batch.py:894. Budget enforcement now "
            "fires for requests without penalty params. NULL on Genesis "
            "PROD (we don't enable thinking_token_budget); defensive for "
            "operators who experiment. Backport of vllm#41674 (JasonKeyiL, "
            "OPEN at backport time)."
        ),
        patch_name=patcher.patch_name,
    )
