# SPDX-License-Identifier: Apache-2.0
"""TDD for PR41674 (vllm#41674) — direct upstream PR backport.

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
        assert "PR41674" in PATCH_REGISTRY

    def test_metadata(self):
        from vllm._genesis.dispatcher import PATCH_REGISTRY
        meta = PATCH_REGISTRY["PR41674"]
        assert meta["env_flag"] == "GENESIS_ENABLE_PR41674"
        assert meta["default_on"] is False
        assert meta["category"] == "stability"
        assert meta["upstream_pr"] == 41674

    def test_apply_skipped_when_env_disabled(self, monkeypatch):
        from vllm._genesis.wiring.perf_hotfix import patch_pr41674_thinking_budget_inverted_bool as p
        monkeypatch.delenv("GENESIS_ENABLE_PR41674", raising=False)
        status, reason = p.apply()
        assert status == "skipped"

    def test_anchor_constants_present(self):
        from vllm._genesis.wiring.perf_hotfix import patch_pr41674_thinking_budget_inverted_bool as p
        assert p.PR41674_OLD
        assert p.PR41674_NEW
        # Sanity: removes the inverted `not`
        assert "or not thinking_budget_tracks_reqs" in p.PR41674_OLD
        assert "or not thinking_budget_tracks_reqs" not in p.PR41674_NEW
        assert "or thinking_budget_tracks_reqs" in p.PR41674_NEW


class TestApplyAllWiring:
    """Verify the patch is wired into apply_all dispatcher."""

    def test_apply_patch_n67_function_exists(self):
        # PR41674 collapsed into the metadata-driven executor (2026-06): assert
        # the registry seam, not a scaffolding function name.
        import vllm._genesis.patches.apply_all  # noqa: F401  (triggers wiring-bind)
        from vllm._genesis.dispatcher import PATCH_REGISTRY
        entry = PATCH_REGISTRY["PR41674"]
        assert entry["wiring"] == "patch_pr41674_thinking_budget_inverted_bool"
        assert callable(entry.get("apply_callable"))
