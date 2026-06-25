# SPDX-License-Identifier: Apache-2.0
"""TDD for PR41418b — TQ centroids disk-persistent cache."""
from __future__ import annotations

import pytest


def _wiring():
    from vllm._genesis.wiring.perf_hotfix import patch_pr41418b_tq_centroids_disk_cache as M
    return M


def test_anchor_targets_get_centroids_body():
    M = _wiring()
    assert "@lru_cache(maxsize=32)" in M.ANCHOR_OLD
    assert "def get_centroids(d: int, bits: int)" in M.ANCHOR_OLD
    assert "centroids, _ = solve_lloyd_max(d, bits)" in M.ANCHOR_OLD


def test_replacement_has_disk_cache_logic():
    M = _wiring()
    assert "PR41418b" in M.ANCHOR_NEW
    assert "_GENESIS_PR41418b_CACHE_PATH" in M.ANCHOR_NEW
    assert "pickle" in M.ANCHOR_NEW.lower() or "pickle.dump" in M.ANCHOR_NEW
    assert "tempfile" in M.ANCHOR_NEW
    assert "os.replace" in M.ANCHOR_NEW  # atomic rename
    assert "@lru_cache(maxsize=32)" in M.ANCHOR_NEW  # in-memory still works


def test_replacement_defensive_fallthrough():
    M = _wiring()
    # Must fall through to solver on cache failure
    assert "except Exception:" in M.ANCHOR_NEW
    # Must call solver as final fallback
    assert "centroids, _ = solve_lloyd_max(d, bits)" in M.ANCHOR_NEW


def test_idempotent_on_synthetic(tmp_path):
    """Audit A-16 fix: synthetic target NO LONGER includes `import os` —
    proves PR41418b replacement self-contains all `os` usage in local imports."""
    from vllm._genesis.wiring.text_patch import (
        TextPatch, TextPatcher, TextPatchResult,
    )
    M = _wiring()
    target = tmp_path / "centroids.py"
    # NOTE: `import os` deliberately ABSENT — this is what reveals A-02 bug
    # in the original PR41418b. After A-02 fix, replacement uses local `import os as _os`
    # inside helper bodies, so module-level os dependency is removed.
    target.write_text("from functools import lru_cache\n" + M.ANCHOR_OLD + "\n")
    patcher = TextPatcher(
        patch_name="PR41418b test",
        target_file=str(target),
        marker=M.GENESIS_PR41418b_MARKER,
        sub_patches=[TextPatch(name="pn57", anchor=M.ANCHOR_OLD,
                                replacement=M.ANCHOR_NEW, required=True)],
    )
    r1, _ = patcher.apply()
    assert r1 == TextPatchResult.APPLIED
    body1 = target.read_text()
    assert "PR41418b" in body1
    # CRITICAL audit A-02 invariant: post-patch body must NOT have module-level
    # `os.<anything>` calls — only `import os as _os` inside function bodies.
    # Module-level `os.path.expanduser(...)` would crash on import.
    lines = body1.splitlines()
    in_function = False
    for line in lines:
        stripped = line.strip()
        # crude function-body detection: `def ...:` increases depth
        if stripped.startswith("def ") and stripped.endswith(":"):
            in_function = True
            continue
        if line and not line.startswith((" ", "\t")) and stripped:
            in_function = False
        # at module level, must NOT have bare `os.<...>` or `os.path...`
        if not in_function and (stripped.startswith("os.") or
                                 " os." in stripped and "import os" not in stripped):
            # only acceptable as comment
            if not stripped.startswith("#"):
                pytest.fail(f"Module-level os.* usage detected (A-02 violation): {line!r}")
    r2, _ = patcher.apply()
    assert r2 == TextPatchResult.IDEMPOTENT


def test_a02_replacement_no_module_level_os_calls():
    """Audit A-02 explicit gate: ANCHOR_NEW must not have module-level os.* calls."""
    M = _wiring()
    # All os.* references must be inside function bodies (lines starting with whitespace)
    for line in M.ANCHOR_NEW.splitlines():
        if "os." in line and "import os" not in line and not line.strip().startswith("#"):
            assert line.startswith((" ", "\t")), (
                f"Module-level os.* found in ANCHOR_NEW: {line!r} — "
                "violates audit A-02 invariant"
            )


def test_env_flag_default_off(monkeypatch):
    from vllm._genesis.dispatcher import should_apply
    monkeypatch.delenv("GENESIS_ENABLE_PR41418B_TQ_CENTROIDS_DISK_CACHE", raising=False)
    decision, _ = should_apply("PR41418b")
    assert decision is False


def test_env_flag_engages(monkeypatch):
    from vllm._genesis.dispatcher import should_apply
    monkeypatch.setenv("GENESIS_ENABLE_PR41418B_TQ_CENTROIDS_DISK_CACHE", "1")
    decision, _ = should_apply("PR41418b")
    assert decision is True


def test_registry_entry_complete():
    from vllm._genesis.dispatcher import PATCH_REGISTRY
    assert "PR41418b" in PATCH_REGISTRY
    meta = PATCH_REGISTRY["PR41418b"]
    assert meta["upstream_pr"] == 41418
    assert "centroids" in meta["title"].lower()


def test_apply_all_registers_pn57():
    from vllm._genesis.patches import apply_all
    # Collapsed to the metadata-driven executor: assert the registry seam,
    # not a scaffolding function name. (apply_all import above bound apply_callable.)
    from vllm._genesis.dispatcher import PATCH_REGISTRY
    entry = PATCH_REGISTRY["PR41418b"]
    assert entry["wiring"] == "patch_pr41418b_tq_centroids_disk_cache" and callable(entry.get("apply_callable"))