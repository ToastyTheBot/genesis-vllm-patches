# SPDX-License-Identifier: Apache-2.0
"""TDD for PR44283 — Anthropic system-role messages inside messages array (vllm#44283)."""
from __future__ import annotations

import os
from pathlib import Path

import pytest


def _wiring():
    from vllm._genesis.wiring.middleware import (
        patch_pr44283_anthropic_system_role as M,
    )
    return M


# ── Anchor / replacement shape ──────────────────────────────────────────────
def test_anchor_protocol_widens_role_literal():
    M = _wiring()
    assert 'role: Literal["user", "assistant"]' in M.PROTOCOL_OLD
    assert 'role: Literal["user", "assistant", "system"]' in M.PROTOCOL_NEW
    # The only change is the added "system" member — content line preserved.
    assert "content: str | list[AnthropicContentBlock]" in M.PROTOCOL_OLD
    assert "content: str | list[AnthropicContentBlock]" in M.PROTOCOL_NEW


def test_anchor_serving_system_message_collects_both_sources():
    M = _wiring()
    # Old: early-return + single-source top-level system handling.
    assert "if not anthropic_request.system:" in M.SERVING_SYS_OLD
    assert "system_prompt" in M.SERVING_SYS_OLD
    # New: accumulate into system_parts from BOTH top-level + messages array.
    assert "system_parts" in M.SERVING_SYS_NEW
    assert "for msg in anthropic_request.messages:" in M.SERVING_SYS_NEW
    assert 'if msg.role != "system":' in M.SERVING_SYS_NEW
    assert '"".join(system_parts)' in M.SERVING_SYS_NEW


def test_anchor_serving_convert_messages_skips_system():
    M = _wiring()
    assert "Convert Anthropic messages to OpenAI format" in M.SERVING_MSG_OLD
    assert "# type: ignore" in M.SERVING_MSG_OLD
    assert 'if msg.role == "system":' in M.SERVING_MSG_NEW
    assert "continue" in M.SERVING_MSG_NEW


def test_replacements_carry_pr44283_marker():
    M = _wiring()
    for name, new in [
        ("SERVING_SYS_NEW", M.SERVING_SYS_NEW),
        ("SERVING_MSG_NEW", M.SERVING_MSG_NEW),
    ]:
        assert "PR44283" in new, f"{name} missing PR44283 marker"


# ── Idempotency (temp files, no real vLLM needed) ───────────────────────────
def test_idempotent_protocol(tmp_path):
    from vllm._genesis.wiring.text_patch import (
        TextPatch, TextPatcher, TextPatchResult,
    )
    M = _wiring()
    target = tmp_path / "protocol.py"
    target.write_text("# header\n" + M.PROTOCOL_OLD + "\n")
    patcher = TextPatcher(
        patch_name="PR44283 protocol test",
        target_file=str(target),
        marker=M.GENESIS_PR44283_MARKER + " (protocol)",
        sub_patches=[TextPatch(name="pr44283_role_literal",
                               anchor=M.PROTOCOL_OLD,
                               replacement=M.PROTOCOL_NEW, required=True)],
    )
    r1, _ = patcher.apply()
    assert r1 == TextPatchResult.APPLIED
    assert 'role: Literal["user", "assistant", "system"]' in target.read_text()
    r2, _ = patcher.apply()
    assert r2 == TextPatchResult.IDEMPOTENT


def test_idempotent_serving_system_message(tmp_path):
    from vllm._genesis.wiring.text_patch import (
        TextPatch, TextPatcher, TextPatchResult,
    )
    M = _wiring()
    target = tmp_path / "serving.py"
    target.write_text("# header\n" + M.SERVING_SYS_OLD + "\n# tail\n")
    patcher = TextPatcher(
        patch_name="PR44283 serving sys test",
        target_file=str(target),
        marker=M.GENESIS_PR44283_MARKER + " (serving)",
        sub_patches=[TextPatch(name="pr44283_convert_system_message",
                               anchor=M.SERVING_SYS_OLD,
                               replacement=M.SERVING_SYS_NEW, required=True)],
    )
    r1, _ = patcher.apply()
    assert r1 == TextPatchResult.APPLIED
    r2, _ = patcher.apply()
    assert r2 == TextPatchResult.IDEMPOTENT


# ── Anchors exist verbatim in the pinned vLLM source (skips if unavailable) ──
def _pinned_anthropic_dir() -> Path | None:
    candidates = []
    env = os.environ.get("GENESIS_VLLM_PIN_PATH")
    if env:
        candidates.append(Path(env))
    candidates += [Path("/tmp/vllm/vllm"), Path("/root/vllm/vllm"),
                   Path("/tmp/vllm_pin/vllm")]
    for root in candidates:
        d = root / "entrypoints" / "anthropic"
        if (d / "serving.py").exists() and (d / "protocol.py").exists():
            return d
    return None


def test_anchors_present_in_pinned_source():
    d = _pinned_anthropic_dir()
    if d is None:
        pytest.skip("pinned vLLM Anthropic endpoint source not available")
    M = _wiring()
    proto = (d / "protocol.py").read_text()
    serv = (d / "serving.py").read_text()
    assert proto.count(M.PROTOCOL_OLD) == 1, "protocol anchor missing/ambiguous"
    assert serv.count(M.SERVING_SYS_OLD) == 1, "serving system-message anchor missing/ambiguous"
    assert serv.count(M.SERVING_MSG_OLD) == 1, "serving convert-messages anchor missing/ambiguous"


# ── Gating + registry seam ──────────────────────────────────────────────────
def test_env_flag_default_off(monkeypatch):
    from vllm._genesis.dispatcher import should_apply
    monkeypatch.delenv("GENESIS_ENABLE_PR44283_ANTHROPIC_SYSTEM_ROLE", raising=False)
    decision, _ = should_apply("PR44283")
    assert decision is False


def test_env_flag_engages(monkeypatch):
    from vllm._genesis.dispatcher import should_apply
    monkeypatch.setenv("GENESIS_ENABLE_PR44283_ANTHROPIC_SYSTEM_ROLE", "1")
    decision, _ = should_apply("PR44283")
    assert decision is True


def test_registry_entry_complete():
    from vllm._genesis.dispatcher import PATCH_REGISTRY
    assert "PR44283" in PATCH_REGISTRY
    meta = PATCH_REGISTRY["PR44283"]
    assert meta["upstream_pr"] == 44283
    assert meta["env_flag"] == "GENESIS_ENABLE_PR44283_ANTHROPIC_SYSTEM_ROLE"
    assert meta["default_on"] is False


def test_registry_seam_wired():
    # Importing apply_all binds the metadata-driven executor onto the entry.
    from vllm._genesis.patches import apply_all  # noqa: F401
    from vllm._genesis.dispatcher import PATCH_REGISTRY
    entry = PATCH_REGISTRY["PR44283"]
    assert entry["wiring"] == "patch_pr44283_anthropic_system_role"
    assert callable(entry.get("apply_callable"))
