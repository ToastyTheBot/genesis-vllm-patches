# SPDX-License-Identifier: Apache-2.0
"""TDD for PN67 (vllm#41674) — direct upstream PR backport.

Verifies:
- Registers correctly in PATCH_REGISTRY
- Correct env_flag + opt-in default
- apply() returns "skipped" when env disabled
- Unique anchors visible in current pin (best-effort)

Note: PN66 (vllm#41696) was removed in Phase B — upstream closed that PR
(see PLAN.md). This file's PN66 coverage was dropped with it.
"""
from __future__ import annotations


class TestPN67Registration:
    def test_in_registry(self):
        from vllm._genesis.dispatcher import PATCH_REGISTRY
        assert "PN67" in PATCH_REGISTRY

    def test_metadata(self):
        from vllm._genesis.dispatcher import PATCH_REGISTRY
        meta = PATCH_REGISTRY["PN67"]
        assert meta["env_flag"] == "GENESIS_ENABLE_PN67"
        assert meta["default_on"] is False
        assert meta["category"] == "stability"
        assert meta["upstream_pr"] == 41674

    def test_apply_skipped_when_env_disabled(self, monkeypatch):
        from vllm._genesis.wiring.perf_hotfix import patch_N67_thinking_budget_inverted_bool as p
        monkeypatch.delenv("GENESIS_ENABLE_PN67", raising=False)
        status, reason = p.apply()
        assert status == "skipped"

    def test_anchor_constants_present(self):
        from vllm._genesis.wiring.perf_hotfix import patch_N67_thinking_budget_inverted_bool as p
        assert p.PN67_OLD
        assert p.PN67_NEW
        # Sanity: removes the inverted `not`
        assert "or not thinking_budget_tracks_reqs" in p.PN67_OLD
        assert "or not thinking_budget_tracks_reqs" not in p.PN67_NEW
        assert "or thinking_budget_tracks_reqs" in p.PN67_NEW


class TestApplyAllWiring:
    """Verify the patch is wired into apply_all dispatcher."""

    def test_apply_patch_n67_function_exists(self):
        from vllm._genesis.patches.apply_all import apply_patch_N67_thinking_budget_inverted_bool
        assert callable(apply_patch_N67_thinking_budget_inverted_bool)
