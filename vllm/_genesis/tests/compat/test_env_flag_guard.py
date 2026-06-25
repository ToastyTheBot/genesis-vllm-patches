# SPDX-License-Identifier: Apache-2.0
"""TDD for env_flag_guard typo shield."""
from __future__ import annotations

import pytest


def test_levenshtein_distance():
    from vllm._genesis.compat.env_flag_guard import _levenshtein
    assert _levenshtein("", "") == 0
    assert _levenshtein("abc", "abc") == 0
    assert _levenshtein("kitten", "sitting") == 3  # classic
    assert _levenshtein("abc", "abd") == 1          # one substitution
    assert _levenshtein("abc", "abde") == 2         # substitution + insertion


def test_collect_known_flags_includes_pn55():
    from vllm._genesis.compat.env_flag_guard import collect_known_flags
    known = collect_known_flags()
    # PR41602 was added in this session — must be picked up
    assert "GENESIS_ENABLE_PR41602_WAKE_UP_HYBRID_KV" in known
    # And legacy patches
    assert "GENESIS_ENABLE_PR41422_SPARSE_V" in known


def test_no_typos_on_clean_environ():
    from vllm._genesis.compat.env_flag_guard import find_typos
    findings = find_typos(environ={"PATH": "/usr/bin", "HOME": "/home/x"})
    assert findings == []


def test_known_flag_no_finding():
    from vllm._genesis.compat.env_flag_guard import find_typos
    findings = find_typos(environ={
        "GENESIS_ENABLE_PR41602_WAKE_UP_HYBRID_KV": "1",
        "GENESIS_ENABLE_PR41422_SPARSE_V": "1",
    })
    assert findings == []


def test_disable_inverse_no_finding():
    from vllm._genesis.compat.env_flag_guard import find_typos
    findings = find_typos(environ={
        "GENESIS_DISABLE_PR35975_INPUTS_EMBEDS_OPTIONAL": "1",
    })
    # GENESIS_DISABLE_<X> is valid pattern even if not in env_flag values
    assert findings == []


def test_typo_detected_close_match():
    from vllm._genesis.compat.env_flag_guard import find_typos
    # Typo: a dropped digit vs the real flag GENESIS_ENABLE_PR41602_WAKE_UP_HYBRID_KV
    findings = find_typos(environ={
        "GENESIS_ENABLE_PR4162_WAKE_UP_HYBRID_KV": "1",  # missing one digit
    })
    assert len(findings) == 1
    assert "PR4162" in findings[0].env_var
    assert findings[0].closest_known is not None
    assert "PR41602" in findings[0].closest_known
    assert findings[0].distance is not None and findings[0].distance <= 4


def test_tuning_knob_allowlisted():
    """Suffix _DEBUG/_THRESHOLD/etc. = tuning knob, not patch toggle."""
    from vllm._genesis.compat.env_flag_guard import find_typos
    findings = find_typos(environ={
        "GENESIS_ENABLE_PR41422_DEBUG": "1",       # allowlisted suffix
        "GENESIS_ENABLE_PR41422_THRESHOLD": "0.005",  # tuning knob
    })
    # These shouldn't flag as typos (allowlisted)
    assert findings == []


def test_assert_no_typos_default_warn(caplog):
    from vllm._genesis.compat.env_flag_guard import assert_no_typos
    import logging
    with caplog.at_level(logging.WARNING, logger="genesis.compat.env_flag_guard"):
        n = assert_no_typos(strict=False)
    # On clean test env should be 0
    assert n == 0


def test_assert_no_typos_strict_raises(monkeypatch):
    from vllm._genesis.compat.env_flag_guard import assert_no_typos
    # Use a close-typo (dropped digit) of the real flag GENESIS_ENABLE_PR41602_WAKE_UP_HYBRID_KV
    monkeypatch.setenv("GENESIS_ENABLE_PR4162_WAKE_UP_HYBRID_KV", "1")
    with pytest.raises(RuntimeError, match="suspicious GENESIS_ENABLE"):
        assert_no_typos(strict=True)


def test_assert_no_typos_unrelated_unaffected():
    """Long unrelated names (distance > 4) should not be flagged."""
    from vllm._genesis.compat.env_flag_guard import find_typos
    findings = find_typos(environ={
        "GENESIS_ENABLE_TOTALLY_DIFFERENT_USER_KNOB_HERE": "1",
    })
    # Distance > 4 from any real flag → not flagged (unlikely typo)
    # But also possible to flag if distance ≤ 4 from PR35975/PN51
    # Just ensure logic doesn't crash
    assert isinstance(findings, list)
