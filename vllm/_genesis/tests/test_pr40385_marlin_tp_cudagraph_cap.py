# SPDX-License-Identifier: Apache-2.0
"""TDD tests for PR40385 — Marlin TP cudagraph cap on Ampere (vllm#40385 backport).

Pure-Python tests on the text-patch generator + dispatcher entry. No vLLM
runtime dependency — exercises anchor structure, replacement invariants,
marker versioning, and apply() decision tree.

Covers:
  - marker is versioned (v7.62.12) + references upstream PR
  - both anchors non-empty + replacements differ
  - both anchors carry enough context for uniqueness (>= 80 chars)
  - replacement preserves exact gate condition
  - replacement uses _genesis_p95_platform alias to avoid name clash
  - replacement caps to 8 ONLY when user has NOT set custom sizes
  - cap respects existing max (min(current, 8) — not blindly assign 8)
  - apply() short-circuits when env unset
  - dispatcher entry exists and references upstream PR 40385
"""
from __future__ import annotations

import re

import pytest

from vllm._genesis.wiring.hybrid.patch_pr40385_marlin_tp_cudagraph_cap import (
    GENESIS_PR40385_MARKER,
    PR40385_CAP_NEW,
    PR40385_CAP_OLD,
    PR40385_IMPORT_NEW,
    PR40385_IMPORT_OLD,
    _make_patcher,
    apply,
)


# ─── Marker invariants ──────────────────────────────────────────────────


def test_p95_marker_versioned():
    """Marker should embed v7.62.12 + reference upstream PR."""
    assert "v7.62.12" in GENESIS_PR40385_MARKER, (
        f"PR40385 marker {GENESIS_PR40385_MARKER!r} should embed v7.62.12 version tag"
    )
    assert "vllm#40385" in GENESIS_PR40385_MARKER, (
        "PR40385 marker should reference upstream PR for drift detection"
    )


# ─── Anchor / replacement integrity ──────────────────────────────────────


@pytest.mark.parametrize("old,new,label", [
    (PR40385_IMPORT_OLD, PR40385_IMPORT_NEW, "import"),
    (PR40385_CAP_OLD, PR40385_CAP_NEW, "cap_block"),
])
def test_p95_anchors_nonempty_and_replacements_differ(old, new, label):
    assert old.strip(), f"{label}: anchor is empty"
    assert new.strip(), f"{label}: replacement is empty"
    assert old != new, f"{label}: replacement equals anchor (no-op)"


@pytest.mark.parametrize("old,label", [
    (PR40385_IMPORT_OLD, "import"),
    (PR40385_CAP_OLD, "cap_block"),
])
def test_p95_anchors_have_enough_context(old, label):
    """Anchors must be >= 80 chars to be unique against config/vllm.py."""
    assert len(old) >= 80, (
        f"{label}: anchor too short ({len(old)} chars). Risk of multi-match"
    )


@pytest.mark.parametrize("new,label", [
    (PR40385_IMPORT_NEW, "import"),
    (PR40385_CAP_NEW, "cap_block"),
])
def test_p95_replacements_carry_genesis_breadcrumb(new, label):
    """Drift detection requires `[Genesis PR40385` in every modified region."""
    assert "[Genesis PR40385" in new, (
        f"{label}: replacement missing `[Genesis PR40385` breadcrumb"
    )


# ─── Semantic invariants ─────────────────────────────────────────────────


def test_p95_import_uses_alias():
    """The `current_platform` import uses an alias to avoid clobbering
    any existing `current_platform` import elsewhere in vllm.py."""
    assert "_genesis_p95_platform" in PR40385_IMPORT_NEW, (
        "import must use _genesis_p95_platform alias to avoid name clash"
    )
    assert "from vllm.platforms import current_platform as _genesis_p95_platform" in PR40385_IMPORT_NEW


def test_p95_cap_gates_on_all_5_conditions():
    """The cap must check ALL 5 conditions before triggering:
      1. cudagraph_capture_sizes is None
      2. max_cudagraph_capture_size is None
      3. tensor_parallel_size > 1
      4. is_cuda()
      5. is_device_capability_family(80)
    """
    assert "cudagraph_capture_sizes is None" in PR40385_CAP_NEW
    assert "max_cudagraph_capture_size is None" in PR40385_CAP_NEW
    assert "tensor_parallel_size > 1" in PR40385_CAP_NEW
    assert "_genesis_p95_platform.is_cuda()" in PR40385_CAP_NEW
    assert "_genesis_p95_platform.is_device_capability_family(80)" in PR40385_CAP_NEW


def test_p95_cap_only_for_marlin_quants():
    """The cap must check `quantization.endswith('_marlin')` so non-Marlin
    quants (FP8 — our PROD) are unaffected."""
    assert "endswith('_marlin')" in PR40385_CAP_NEW, (
        "cap must only fire for *_marlin quantizations"
    )


def test_p95_cap_uses_min_not_assign():
    """Critical: the cap must `min(max_cudagraph_capture_size, 8)`, NOT
    blindly assign 8. If user-derived value is already < 8, we keep it."""
    assert "min(max_cudagraph_capture_size, 8)" in PR40385_CAP_NEW, (
        "cap must use min() to respect lower existing values"
    )


def test_p95_cap_logs_warning_only_above_8():
    """The warning_once should fire ONLY when we're actually capping
    (max > 8). If max already <= 8, no warning."""
    # Look for the conditional warning
    assert re.search(
        r"if max_cudagraph_capture_size > 8:.*?logger\.warning_once",
        PR40385_CAP_NEW,
        re.DOTALL,
    ), "warning_once must be inside `if max_cudagraph_capture_size > 8:`"


def test_p95_cap_preserves_assert():
    """The downstream assert must still be reachable — the cap inserts
    BEFORE the assert, doesn't remove it."""
    assert "assert max_cudagraph_capture_size >= 1" in PR40385_CAP_NEW, (
        "post-cap assert must still be in the new replacement (we anchor on it)"
    )


# ─── Dispatcher integration ──────────────────────────────────────────────


def test_p95_in_PATCH_REGISTRY():
    from vllm._genesis.dispatcher import PATCH_REGISTRY
    assert "PR40385" in PATCH_REGISTRY, "PR40385 must be registered in PATCH_REGISTRY"
    p = PATCH_REGISTRY["PR40385"]
    assert p["env_flag"] == "GENESIS_ENABLE_PR40385_MARLIN_TP_CUDAGRAPH_CAP"
    assert p["default_on"] is False, "PR40385 must be opt-in (default OFF)"
    assert p["upstream_pr"] == 40385


def test_p95_dispatcher_quant_format_includes_marlin_paths():
    """applies_to must include all quant_formats that route through Marlin."""
    from vllm._genesis.dispatcher import PATCH_REGISTRY
    af = PATCH_REGISTRY["PR40385"].get("applies_to", {}).get("quant_format", [])
    # Lorbus int4 = autoround_int4 → Marlin
    # Minachist gs128 = autoround_int8 → Marlin
    assert "autoround_int4" in af, "Lorbus INT4 quant_format must be in applies_to"
    assert "autoround_int8" in af, "Minachist gs128 quant_format must be in applies_to"


# ─── Patcher structure ──────────────────────────────────────────────────


def test_p95_patcher_has_two_required_sub_patches():
    patcher = _make_patcher()
    if patcher is None:
        pytest.skip("vllm not installed locally")
    assert len(patcher.sub_patches) == 2
    for sp in patcher.sub_patches:
        assert sp.required, f"sub-patch {sp.name!r} must be required"
    names = {sp.name for sp in patcher.sub_patches}
    assert names == {"p95_import_current_platform", "p95_marlin_cap_block"}


# ─── apply() short-circuits ─────────────────────────────────────────────


def test_p95_apply_skipped_when_env_unset(monkeypatch):
    monkeypatch.delenv("GENESIS_ENABLE_PR40385_MARLIN_TP_CUDAGRAPH_CAP", raising=False)
    status, reason = apply()
    assert status == "skipped"
    assert "PR40385" in reason or "default_on" in reason or "env_flag" in reason
