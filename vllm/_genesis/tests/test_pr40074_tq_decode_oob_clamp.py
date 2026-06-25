# SPDX-License-Identifier: Apache-2.0
"""Unit tests for PR40074 — TQ decode IOOB safe_page_idx clamp (vllm#40074).

What we can test on a CPU dev host (no Triton execution):
  1. Anchor matches a synthetic vanilla snippet of `_tq_decode_stage1`.
  2. Replacement adds `safe_page_idx = tl.where(...)` exactly once.
  3. Marker is unique and idempotency works (re-applying is no-op).
  4. Drift markers fire correctly:
       a) Genesis marker present → skip with "already patched"
       b) Upstream `safe_page_idx = tl.where(kv_mask, page_idx, 0)` present
          → skip with drift detected (upstream merged the fix)
  5. Anchor missing → skip (file unrecognized / pin drift).
  6. Dispatcher integration: env unset → SKIP, env=1 → APPLY.

We deliberately do NOT compile Triton or test the GPU kernel — that's
covered by upstream's CI on the merged fix; our concern is the text-patch
discipline, anchor stability, drift detection, and marker uniqueness.
"""
from __future__ import annotations

import os

import pytest


# ─── Synthetic baseline matching upstream pre-#40074 source ────────────


_PR40074_VANILLA_SNIPPET = """# SPDX-License-Identifier: Apache-2.0
# Synthetic baseline of vllm/v1/attention/ops/triton_turboquant_decode.py

import triton.language as tl

@triton.jit
def _tq_decode_stage1(
    Block_table_ptr, KV_cache_ptr, ...
):
    for start_n in range(split_start, split_end, BLOCK_KV):
        kv_offs = start_n + kv_range
        kv_mask = kv_offs < split_end

        page_idx = kv_offs // BLOCK_SIZE
        page_off = kv_offs % BLOCK_SIZE
        block_nums = tl.load(
            Block_table_ptr + bt_base + page_idx,
            mask=kv_mask,
            other=0,
        ).to(tl.int64)

        slot_bases = (
            block_nums * stride_cache_block
            + page_off.to(tl.int64) * stride_cache_pos
        )
"""


_PR40074_UPSTREAM_MERGED = """# Synthetic upstream-merged variant (after #40074 lands)

import triton.language as tl

@triton.jit
def _tq_decode_stage1(...):
    for start_n in range(split_start, split_end, BLOCK_KV):
        kv_offs = start_n + kv_range
        kv_mask = kv_offs < split_end

        page_idx = kv_offs // BLOCK_SIZE
        page_off = kv_offs % BLOCK_SIZE
        # Clamp OOB lanes to index 0 before pointer arithmetic so Triton's
        # bounds checker does not fire on masked-out lanes
        safe_page_idx = tl.where(kv_mask, page_idx, 0)
        block_nums = tl.load(
            Block_table_ptr + bt_base + safe_page_idx,
            mask=kv_mask,
            other=0,
        ).to(tl.int64)
"""


# ─── Test fixture: patch resolver to point at synthetic file ───────────


@pytest.fixture
def fake_tq_decode(tmp_path, monkeypatch):
    """Write synthetic vanilla source to tmp file and redirect the
    resolver to point at it so apply() operates on the synthetic file."""
    path = tmp_path / "triton_turboquant_decode.py"
    path.write_text(_PR40074_VANILLA_SNIPPET)

    from vllm._genesis.wiring.kernels import patch_pr40074_tq_decode_oob_clamp as p14
    monkeypatch.setattr(
        p14, "resolve_vllm_file",
        lambda rel: str(path) if "triton_turboquant_decode" in rel else None,
    )
    monkeypatch.setattr(p14, "vllm_install_root", lambda: "/fake/install/root")
    return path


@pytest.fixture
def fake_tq_decode_already_upstream(tmp_path, monkeypatch):
    """Synthetic source that ALREADY has the upstream fix merged."""
    path = tmp_path / "triton_turboquant_decode.py"
    path.write_text(_PR40074_UPSTREAM_MERGED)

    from vllm._genesis.wiring.kernels import patch_pr40074_tq_decode_oob_clamp as p14
    monkeypatch.setattr(
        p14, "resolve_vllm_file",
        lambda rel: str(path) if "triton_turboquant_decode" in rel else None,
    )
    monkeypatch.setattr(p14, "vllm_install_root", lambda: "/fake/install/root")
    return path


@pytest.fixture
def env_pn14_on(monkeypatch):
    """Enable PR40074 via env flag for these tests."""
    monkeypatch.setenv("GENESIS_ENABLE_PR40074_TQ_DECODE_OOB_CLAMP", "1")
    yield


# ─── Anchor / replacement structural tests ─────────────────────────────


class TestPn14AnchorInvariants:
    """Anchor-invariant guards. If a refactor changes the anchor or the
    replacement, these tests force the developer to reckon with it.
    """

    def test_anchor_matches_synthetic_vanilla(self):
        from vllm._genesis.wiring.kernels.patch_pr40074_tq_decode_oob_clamp import (
            PR40074_ANCHOR,
        )
        assert PR40074_ANCHOR in _PR40074_VANILLA_SNIPPET, (
            "PR40074 anchor no longer matches the synthetic vanilla snippet — "
            "either upstream changed the kernel signature, or our anchor was "
            "edited. Re-derive from current upstream source."
        )

    def test_replacement_introduces_safe_page_idx(self):
        from vllm._genesis.wiring.kernels.patch_pr40074_tq_decode_oob_clamp import (
            PR40074_REPLACEMENT,
        )
        assert "safe_page_idx" in PR40074_REPLACEMENT
        assert "tl.where(kv_mask, page_idx, 0)" in PR40074_REPLACEMENT

    def test_replacement_uses_safe_idx_in_pointer_arith(self):
        """The replacement must use safe_page_idx (not page_idx) in
        Block_table_ptr arithmetic. Otherwise the patch is a no-op."""
        from vllm._genesis.wiring.kernels.patch_pr40074_tq_decode_oob_clamp import (
            PR40074_REPLACEMENT,
        )
        assert "Block_table_ptr + bt_base + safe_page_idx" in PR40074_REPLACEMENT

    def test_marker_string_is_unique_and_versioned(self):
        from vllm._genesis.wiring.kernels.patch_pr40074_tq_decode_oob_clamp import (
            GENESIS_PR40074_MARKER,
        )
        assert "PR40074" in GENESIS_PR40074_MARKER
        assert "vllm#40074" in GENESIS_PR40074_MARKER

    def test_drift_markers_include_genesis_and_upstream(self):
        """Drift detection must catch BOTH 'we already patched' AND
        'upstream merged the fix'."""
        from vllm._genesis.wiring.kernels.patch_pr40074_tq_decode_oob_clamp import (
            _make_patcher,
        )
        # _make_patcher needs target file resolution; we just inspect
        # the function-level constants.
        from vllm._genesis.wiring.kernels import patch_pr40074_tq_decode_oob_clamp as p14
        # GENESIS_PR40074_MARKER itself (covered above)
        # And the upstream-merged signature should also be in the drift list:
        # Check via re-construction: we expect both markers in upstream_drift
        # of the produced TextPatcher.
        # Patcher requires a real file path; use a tmp file just for shape.
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            f.write(_PR40074_VANILLA_SNIPPET.encode())
            f.flush()
            fpath = f.name
        try:
            # monkeypatch via attribute
            orig = p14.resolve_vllm_file
            try:
                p14.resolve_vllm_file = lambda rel: fpath
                patcher = _make_patcher()
                drift_markers = patcher.upstream_drift_markers
            finally:
                p14.resolve_vllm_file = orig
        finally:
            os.unlink(fpath)
        assert any("Genesis PR40074" in m for m in drift_markers)
        assert any("safe_page_idx = tl.where" in m for m in drift_markers)


# ─── apply() behavior tests ─────────────────────────────────────────────


class TestPn14ApplyBehavior:
    def test_skip_when_env_unset(self, fake_tq_decode):
        """Default OFF — env unset → skip without touching the file."""
        from vllm._genesis.wiring.kernels import patch_pr40074_tq_decode_oob_clamp as p14
        before = fake_tq_decode.read_text()
        status, reason = p14.apply()
        after = fake_tq_decode.read_text()
        assert status == "skipped", f"got {status}, reason={reason}"
        assert before == after, "file must not be modified when env is unset"

    def test_apply_when_env_set(self, fake_tq_decode, env_pn14_on):
        from vllm._genesis.wiring.kernels import patch_pr40074_tq_decode_oob_clamp as p14
        status, reason = p14.apply()
        assert status == "applied", f"expected applied, got {status} ({reason})"
        text = fake_tq_decode.read_text()
        assert "safe_page_idx = tl.where(kv_mask, page_idx, 0)" in text
        assert "Block_table_ptr + bt_base + safe_page_idx" in text
        from vllm._genesis.wiring.kernels.patch_pr40074_tq_decode_oob_clamp import (
            GENESIS_PR40074_MARKER,
        )
        assert GENESIS_PR40074_MARKER in text

    def test_idempotent_reapply(self, fake_tq_decode, env_pn14_on):
        """Re-applying should be a no-op (marker present)."""
        from vllm._genesis.wiring.kernels import patch_pr40074_tq_decode_oob_clamp as p14
        p14.apply()
        first_text = fake_tq_decode.read_text()
        p14.apply()
        second_text = fake_tq_decode.read_text()
        assert first_text == second_text, (
            "re-applying PR40074 changed the file — idempotency broken"
        )

    def test_skip_when_upstream_already_merged(
        self, fake_tq_decode_already_upstream, env_pn14_on
    ):
        """If upstream PR #40074 has merged (vanilla source already has
        `safe_page_idx`), the drift marker fires and we skip."""
        from vllm._genesis.wiring.kernels import patch_pr40074_tq_decode_oob_clamp as p14
        before = fake_tq_decode_already_upstream.read_text()
        status, _reason = p14.apply()
        after = fake_tq_decode_already_upstream.read_text()
        assert status == "skipped"
        assert before == after, (
            "PR40074 must not double-patch a file that already has the "
            "upstream fix. Drift marker should have caught this."
        )


# ─── Dispatcher / registry integration ─────────────────────────────────


class TestPn14DispatcherIntegration:
    def test_pn14_in_registry(self):
        from vllm._genesis.dispatcher import PATCH_REGISTRY
        assert "PR40074" in PATCH_REGISTRY
        meta = PATCH_REGISTRY["PR40074"]
        assert meta.get("env_flag") == "GENESIS_ENABLE_PR40074_TQ_DECODE_OOB_CLAMP"
        assert meta.get("default_on") is False
        assert meta.get("upstream_pr") == 40074
        # Should declare it applies only on TurboQuant configs
        applies_to = meta.get("applies_to", {})
        assert applies_to.get("is_turboquant") == [True]

    def test_pn14_should_apply_default_off(self, monkeypatch):
        """With env unset, dispatcher returns False."""
        monkeypatch.delenv(
            "GENESIS_ENABLE_PR40074_TQ_DECODE_OOB_CLAMP", raising=False,
        )
        from vllm._genesis.dispatcher import should_apply
        decision, _reason = should_apply("PR40074")
        assert decision is False

    def test_pn14_should_apply_env_on(self, monkeypatch):
        """With env=1, dispatcher returns True."""
        monkeypatch.setenv("GENESIS_ENABLE_PR40074_TQ_DECODE_OOB_CLAMP", "1")
        from vllm._genesis.dispatcher import should_apply
        decision, _reason = should_apply("PR40074")
        assert decision is True


# ─── upstream_compat marker ────────────────────────────────────────────


class TestPn14UpstreamCompat:
    def test_pn14_marker_present_in_compat(self):
        """upstream_compat must declare the #40074 marker so other
        tooling knows what symbol to grep for to detect upstream merge."""
        from vllm._genesis.patches.upstream_compat import all_markers
        markers = all_markers()
        # Locate the entry by EXACT key (other entries may mention 40074
        # in their description as cross-references — those don't count).
        found = markers.get("PR_40074_tq_decode_oob_clamp")
        assert found is not None, (
            "PR_40074_tq_decode_oob_clamp key not found in upstream_compat. "
            f"Keys present: {sorted(markers.keys())}"
        )
        assert found.get("marker") == "safe_page_idx", (
            f"marker must point at the new symbol introduced by #40074, "
            f"got {found.get('marker')!r}"
        )
