# SPDX-License-Identifier: Apache-2.0
"""Genesis patches orchestrator — applies all enabled patches with defensive guards.

This module replaces the monolithic `patch_genesis_unified.py` orchestration.
It applies each Genesis patch through the 5-layer defensive guard model:

  Layer 1: File exists           → resolve_vllm_file() → skip if None
  Layer 2: Idempotency marker    → grep target file / module attr → skip if already applied
  Layer 3: Upstream merged       → upstream_compat markers → skip if present
  Layer 4: Vendor/chip compat    → is_nvidia_cuda(), is_sm_at_least() → skip on mismatch
  Layer 5: Model/backend arch    → runtime conditional skip where applicable

Each patch reports one of three outcomes:
  - applied:  The patch was wired into the running process.
  - skipped:  Platform/config means this patch is inapplicable (benign).
  - failed:   Something went wrong (missing anchor, import error, etc.).

Usage
-----
From container entrypoint (docker-compose.staging.yml / .yml):

    entrypoint: ["/bin/bash", "-c"]
    command: |
        python3 -m vllm._genesis.patches.apply_all
        exec vllm serve ...

Or standalone for diagnostics:

    $ python3 -m vllm._genesis.patches.apply_all

Exit codes:
  0 — All patches either applied or skipped cleanly (success)
  1 — At least one patch FAILED (anchor miss, unexpected error)
  2 — Setup error (vllm not importable, etc.)

Author: Sandermage(Sander)-Barzov Aleksandr, Ukraine, Odessa
"""
from __future__ import annotations

import json
import logging
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Callable

log = logging.getLogger("genesis.apply_all")


# ═══════════════════════════════════════════════════════════════════════════
#                          ORCHESTRATION STATE
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class PatchResult:
    """Outcome of a single patch attempt."""
    name: str
    status: str           # "applied" | "skipped" | "failed"
    reason: str = ""      # short explanation


@dataclass
class PatchStats:
    """Accumulates per-run statistics for reporting."""
    results: list[PatchResult] = field(default_factory=list)
    # [Genesis T4.6] compile-watchdog: total apply_all elapsed seconds.
    # Set by run() at end. 0.0 if not measured (e.g. dry-run via CLI).
    compile_elapsed_sec: float = 0.0

    @property
    def applied(self) -> list[PatchResult]:
        return [r for r in self.results if r.status == "applied"]

    @property
    def skipped(self) -> list[PatchResult]:
        return [r for r in self.results if r.status == "skipped"]

    @property
    def failed(self) -> list[PatchResult]:
        return [r for r in self.results if r.status == "failed"]

    @property
    def applied_count(self) -> int:
        return len(self.applied)

    @property
    def skipped_count(self) -> int:
        return len(self.skipped)

    @property
    def failed_count(self) -> int:
        return len(self.failed)

    @property
    def partial_apply_warnings(self) -> list[PatchResult]:
        """Skipped patches whose reason signals a real problem (drift,
        ambiguous anchor, anchor-missing — NOT opt-in-OFF, upstream-merged,
        or platform-mismatch which are all expected).

        Surfaced separately from `skipped_count` so noonghunna's "silent
        skip class" diagnosis (club-3090 discussion #19) is impossible to
        miss in the boot summary. Cliff 8 hardening, v7.65.
        """
        # Reasons that indicate a benign/expected skip
        BENIGN = (
            "opt-in",   # matches "opt-in only", "opt-in:", "opt-in env"
            "default off",
            "upstream_merged",
            "upstream_already",
            "upstream_already_contains",
            "upstream may have absorbed",
            "upstream pr",  # "redundant: upstream PR ..."
            "platform mismatch",
            "platform_skip",
            "config: opt-in",
            "config: opt-out",
            "config: skipped",
            "config: neutral",
            "already applied",
            "marker present",
            "soft_skip",
            "no-op",
            "dry-run",
            "vllm install root not discoverable",
            "target file not resolvable",
            "is_pn",
            "unsupported",
            "not applicable",
            "auto-disabled",
            "auto-skip",
            "deprecated",
            "obsolete",
            "redundant",
            "deferred",
            "incompatible with",  # P7 deferred reason
            "retired",            # explicitly retired patches (P8 → 2026-05-04)
            "kernel disabled",    # P67b when P67 kernel disabled (companion patch design)
            "dispatch unused",    # ditto
        )
        warnings = []
        for r in self.skipped:
            reason_lower = (r.reason or "").lower()
            if not any(b.lower() in reason_lower for b in BENIGN):
                warnings.append(r)
        return warnings

    @property
    def partial_apply_warnings_count(self) -> int:
        return len(self.partial_apply_warnings)

    def summary(self) -> dict[str, Any]:
        return {
            "applied": self.applied_count,
            "skipped": self.skipped_count,
            "failed": self.failed_count,
            "partial_apply_warnings": self.partial_apply_warnings_count,
            "details": {
                "applied": [(r.name, r.reason) for r in self.applied],
                "skipped": [(r.name, r.reason) for r in self.skipped],
                "failed": [(r.name, r.reason) for r in self.failed],
                "partial_apply_warnings": [
                    (r.name, r.reason) for r in self.partial_apply_warnings
                ],
            },
        }

    def __str__(self) -> str:
        base = (
            f"Results: {self.applied_count} applied, "
            f"{self.skipped_count} skipped, {self.failed_count} failed"
        )
        warns = self.partial_apply_warnings_count
        if warns:
            base += f", {warns} ⚠️ partial-apply warning(s)"
        return base


# ═══════════════════════════════════════════════════════════════════════════
#                           PATCH REGISTRY
# ═══════════════════════════════════════════════════════════════════════════

# SINGLE REGISTRY (refactor 2026-06): the one source of truth for patch
# metadata AND execution is `dispatcher.PATCH_REGISTRY`. The `@register_patch`
# decorator attaches each apply function onto its dispatcher entry via the
# Phase 5c `apply_callable` field, plus a `_display_name` and `_apply_order`
# index (boot order == decoration order). `apply_all` keeps NO independent
# registry; the module-level `PATCH_REGISTRY` defined at the bottom of this
# file is a DERIVED, ordered `[(display_name, callable), ...]` view of that
# single registry (consumed by `run()` and a few back-compat importers).
# This is why the apply_all↔dispatcher sync contract (formerly policed by
# `test_apply_all_dispatcher_sync.py`) is gone: a callable can no longer
# exist without a metadata entry, and metadata-only entries simply carry no
# callable.
from vllm._genesis.dispatcher import PATCH_REGISTRY as _META_REGISTRY

_APPLY_PATCH_ID_RE = re.compile(r"^apply_patch_(pr\d+[a-z]?|[NM]?\d+[a-zA-Z]?)(?:_|$)")
_apply_order_counter = 0

# ── Explicit boot order (single source of truth) ────────────────────────────
# Boot/apply order used to be implicit — the *source-line* order of the
# `@register_patch`-decorated functions. Collapsing the 85 text-patch/rebind
# dispatch functions into a metadata-driven executor (2026-06) removed that
# implicit signal, so the order is now declared explicitly here. Both the 22
# hand-written outlier functions (via `register_patch`) and the 85 collapsed
# patches (via `_bind_wiring_patches`) take their `_apply_order` from this
# list. Patches not listed (e.g. third-party plugins) sort after, in
# registration order. To reorder boot, edit this list — nothing else.
_APPLY_ORDER: list[str] = [
    "P8", "P3", "P6", "P15", "P12", "P27", "P34", "P29", "P23", "P4", "P5", "P5b", "P31",
    "P22", "P26", "PN59", "PR40962", "PR41467", "PR41418b", "PR41602", "PN54", "PR41411",
    "PN50", "PN51", "PR36138", "PR40738b", "PR40738", "P63", "PR39598", "P65", "P68", "P70",
    "P67", "PR40819", "P78", "P77", "PR40610", "PR37629", "PN61", "PR41674", "PN70", "PN65",
    "PN62", "PR40925", "P82", "P83", "P84", "PR41127", "P103", "PR41123", "PR40941b",
    "PR40941", "PR41043", "PR40385", "PR40849", "PR39930", "PR41142", "P67c", "PR35975",
    "PN40", "PR40425", "PR37521", "PN32", "PN31", "PN30", "PR41446", "PR34207", "PR39148",
    "P15B", "P38B", "PR41422", "PR41418", "PN25", "PR41235", "PR40074", "PR41268",
    "PR40898", "PR39419", "PR40727", "PN17", "PN16", "P85", "PR25784", "P74", "P72", "P67b",
    "PR39055", "PR40768", "P57", "P56", "P44", "P46", "P7b", "P40", "P39a", "P38", "P37",
    "P36", "P32", "P28", "P7", "P17", "P24", "P14", "P18b", "P20", "P1"
]
_ORDER_INDEX: dict[str, int] = {pid: i for i, pid in enumerate(_APPLY_ORDER)}


def register_patch(name: str, patch_id: str | None = None):
    """Attach a patch's apply function onto its dispatcher metadata entry.

    The patch ID is taken from the explicit ``patch_id`` argument when given
    (used by ``PR#####``-named patches whose ID can't be derived from the
    function name); otherwise it is parsed from the function name
    (``apply_patch_<id>_*`` → ``P<id>``, matching the dispatcher key). The
    callable is stored on the single registry as ``apply_callable`` together
    with ``_display_name`` and ``_apply_order`` (boot order == decoration
    order). If no dispatcher entry exists a minimal stub is created and an
    error logged, so a patch is never silently dropped.
    """
    def decorator(fn: Callable[[], PatchResult]) -> Callable[[], PatchResult]:
        global _apply_order_counter
        if patch_id is not None:
            pid = patch_id
        else:
            m = _APPLY_PATCH_ID_RE.match(fn.__name__)
            if m is None:
                pid = fn.__name__
            elif m.group(1).startswith("pr"):
                pid = "PR" + m.group(1)[2:]        # apply_patch_pr40738b_* -> PR40738b
            else:
                pid = "P" + m.group(1)            # apply_patch_67_* -> P67 (legacy)
        entry = _META_REGISTRY.get(pid)
        if entry is None:
            log.error(
                "[Genesis] apply_patch %r (id=%s) has no dispatcher "
                "PATCH_REGISTRY entry — creating a minimal stub so it still "
                "runs; add metadata in dispatcher.py.", fn.__name__, pid,
            )
            entry = _META_REGISTRY.setdefault(
                pid, {"title": name, "category": "uncategorized"}
            )
        entry["apply_callable"] = fn
        entry["_display_name"] = name
        oi = _ORDER_INDEX.get(pid)
        if oi is None:
            # Unlisted (e.g. third-party plugin) — sort after the declared set,
            # preserving registration order among themselves.
            oi = len(_APPLY_ORDER) + _apply_order_counter
            _apply_order_counter += 1
        entry["_apply_order"] = oi
        return fn
    return decorator


def _applied(name: str, reason: str = "") -> PatchResult:
    return PatchResult(name=name, status="applied", reason=reason)


def _skipped(name: str, reason: str) -> PatchResult:
    return PatchResult(name=name, status="skipped", reason=reason)


def _failed(name: str, reason: str) -> PatchResult:
    return PatchResult(name=name, status="failed", reason=reason)


# ═══════════════════════════════════════════════════════════════════════════
#                       PATCH IMPLEMENTATIONS
# ═══════════════════════════════════════════════════════════════════════════

# Module-level state: are we in dry-run or apply mode for this run?
# Set by run(apply=True/False). Dry-run only diagnoses; apply performs the
# actual text-patch / monkey-patch wiring.
_APPLY_MODE: bool = False


_WIRING_STEM_INDEX: dict[str, str] | None = None


def _resolve_wiring_module(stem: str) -> str:
    """Resolve a bare wiring filename stem (e.g. 'patch_67_tq_multi_query_kernel')
    to its full dotted module path. Walks `wiring/` recursively so the
    legacy flat layout AND post-Phase-2.1 category subdirs both work
    transparently.
    """
    global _WIRING_STEM_INDEX
    if _WIRING_STEM_INDEX is None:
        from pathlib import Path
        wiring_dir = Path(__file__).resolve().parent.parent / "wiring"
        idx: dict[str, str] = {}
        if wiring_dir.is_dir():
            for f in wiring_dir.rglob("patch_*.py"):
                rel_parts = f.relative_to(
                    wiring_dir.parent.parent.parent
                ).parts
                idx[f.stem] = ".".join(list(rel_parts[:-1]) + [f.stem])
        _WIRING_STEM_INDEX = idx
    # Fallback to flat layout if not in cache (covers a freshly-added file
    # that wasn't there at first-call time).
    return _WIRING_STEM_INDEX.get(
        stem, f"vllm._genesis.wiring.{stem}"
    )


def _wiring_text_patch(name: str, wiring_module_name: str) -> PatchResult:
    """Generic helper for dry-run / live dispatch of a text-patch wiring module."""
    try:
        import importlib
        mod = importlib.import_module(
            _resolve_wiring_module(wiring_module_name)
        )
    except Exception as e:
        return _failed(name, f"wiring import failed: {e}")

    if not _APPLY_MODE:
        return _applied(name, "dry-run: wiring ready (pass apply=True to execute)")

    try:
        status, reason = mod.apply()
    except Exception as e:
        return _failed(name, f"wiring raised (should not happen): {e}")

    if status == "applied":
        return _applied(name, reason)
    if status == "skipped":
        return _skipped(name, reason)
    return _failed(name, reason)


# ── Metadata-driven dispatch for text-patch / rebind wiring patches ─────────
# 85 patches used to each carry a near-identical ~36-line `apply_patch_*`
# function whose only per-patch variation was the wiring module it imports.
# That ceremony is gone: a patch declares its implementation with a `wiring:
# "<stem>"` field in dispatcher.PATCH_REGISTRY, and this one executor runs it.
# The boot-summary label is composed as "<pid> <title>" from the registry.
# Outliers (kernel installs, rebinds, bundled preallocs, hardcoded skips) that
# carry real logic keep their hand-written `apply_patch_*` function below.
def _apply_wiring_entry(pid: str) -> PatchResult:
    """Generic apply step for a registry entry whose `wiring` field names its
    text-patch/rebind module. Delegates to `_wiring_text_patch`."""
    meta = _META_REGISTRY.get(pid, {})
    name = f"{pid} {meta.get('title', pid)}"
    stem = meta.get("wiring")
    if not stem:
        return _failed(name, f"{pid}: no 'wiring' module declared in PATCH_REGISTRY")
    return _wiring_text_patch(name, stem)


def _make_wiring_callable(pid: str) -> Callable[[], PatchResult]:
    return lambda: _apply_wiring_entry(pid)


def _bind_wiring_patches() -> None:
    """Attach the generic executor onto every registry entry that declares a
    `wiring` module. Runs once at import end, after the hand-written outlier
    `@register_patch` functions have bound themselves, and before the derived
    view is built. Mirrors what `register_patch` does for outliers."""
    for pid, meta in _META_REGISTRY.items():
        stem = meta.get("wiring")
        if not stem:
            continue
        meta["apply_callable"] = _make_wiring_callable(pid)
        meta["_display_name"] = f"{pid} {meta.get('title', pid)}"
        meta["_apply_order"] = _ORDER_INDEX.get(pid, len(_APPLY_ORDER))


@register_patch("P8 KV hybrid reporting (per-token capacity)")
def apply_patch_8_kv_hybrid_reporting() -> PatchResult:
    """Patch 8: RETIRED 2026-05-04 — upstream refactored the API.

    Original purpose: closed the 3.76× KV-cache gap on Qwen3.6-35B-A3B by
    excluding Mamba groups from the per-token capacity divisor.

    Retired because vllm 0.20.2rc1.dev9+g01d4d1ad3 refactored
    `_report_kv_cache_config` to call `get_max_concurrency_for_kv_cache_config`,
    which already handles hybrid groups correctly upstream — our text-patch
    anchors no longer match. See dispatcher.py PATCH_REGISTRY entry for the
    diff-analysis lifecycle marker (`lifecycle: retired_2026-05-04`).

    Skipping silently to avoid a DRIFT WARNING in every boot log. The wiring
    file is kept on disk for git-history reference but never invoked.
    """
    name = "P8 KV hybrid reporting (per-token capacity)"
    return _skipped(name, "retired 2026-05-04 (upstream refactor superseded)")


@register_patch("P29 tool parser IndexError guard")
def apply_patch_29_tool_parser_index_guard() -> PatchResult:
    """Patch 29: Defensive IndexError guard in qwen3coder tool parser.

    Historical bug: `self.streamed_args_for_tool[self.current_tool_index]`
    could raise IndexError when the serving layer processed tools faster
    than the parser tracked them. Baseline v7.0 vLLM already contains
    bounded-index guards at the relevant call sites (lines 609-616, 659-666,
    436-438 of qwen3coder_tool_parser.py). This patch VERIFIES upstream
    acceptance and no-ops if the guards are already in place.

    Scope: the guard we would add is already present in the baseline image
    via upstream PRs. The patch remains registered so that future vLLM
    upgrades where the guard regresses are automatically re-applied.
    """
    name = "P29 tool parser IndexError guard"
    try:
        from vllm._genesis.guards import resolve_vllm_file
    except Exception as e:
        return _failed(name, f"guards import failed: {e}")

    target = resolve_vllm_file("tool_parsers/qwen3coder_tool_parser.py")
    if target is None:
        return _skipped(name, "qwen3coder_tool_parser.py not found")

    try:
        with open(target) as f:
            content = f.read()
    except Exception as e:
        return _skipped(name, f"read_error: {e}")

    # Upstream-merged detection: all three guarded sites must be present.
    has_streamed_guard = (
        "streamed_args_for_tool out of sync" in content
        and "self.current_tool_index < len(self.streamed_args_for_tool)" in content
    )
    has_positions_guard = (
        "if self.current_tool_index >= len(tool_start_positions)" in content
    )

    if has_streamed_guard and has_positions_guard:
        return _applied(
            name,
            "upstream already contains bounded-index guards (no-op)",
        )

    # Baseline image does not have the guards → we would apply them, but for
    # v7.0 the baseline DOES have them, so this path is unreachable on the
    # supported image. Keep the branch for forward-compat.
    return _skipped(
        name,
        "upstream guards absent; text-patch for this regression path not "
        "shipped in v7.0 (reimplement when upstream regresses)",
    )


@register_patch("P23 Marlin FP32_REDUCE env override")
def apply_patch_23_marlin_fp32_reduce() -> PatchResult:
    """Patch 23: NEW in v7.0. Expose `VLLM_MARLIN_FP32_REDUCE` env var plus
    auto-select (disable on SM<90, keep on SM>=90). Kernel-level helper only
    — does NOT yet wire into Marlin launcher (needs upstream coordination or
    additional text-patch on fused_marlin_moe.py)."""
    name = "P23 Marlin FP32_REDUCE env override"
    try:
        from vllm._genesis.kernels.marlin_fp32_reduce import (
            should_disable_fp32_reduce,
            log_decision,
        )
    except Exception as e:
        return _failed(name, f"kernel import failed: {e}")

    if not _APPLY_MODE:
        return _applied(name, "dry-run: kernel helper ready")

    from vllm._genesis.guards import is_nvidia_cuda
    if not is_nvidia_cuda():
        return _skipped(name, "non-NVIDIA — no Marlin path")

    log_decision()  # writes a structured log line
    disabled = should_disable_fp32_reduce()
    return _applied(
        name,
        f"decision: fp32_reduce disabled={disabled} "
        f"(requires upstream wire into Marlin launcher to take effect)",
    )


@register_patch("P5b KV page-size pad-smaller-to-max (env-opt-in)")
def apply_patch_5b_page_size_pad_smaller() -> PatchResult:
    """Patch 5b: pad-SMALLER-to-max KV page-size strategy (alt to P5 v1).

    Frees ~34% per-block VRAM vs P5 v1 LCM-pad-up on Qwen3.6-35B-A3B
    hybrid. Ships env-gated (`GENESIS_ENABLE_P5B=1`) because the
    blast-radius is the KV-cache allocator sizing semantics — operators
    MUST bench GSM8K + long-context regression on VM 100 before
    enabling in prod.

    The precursor attempt (P5 v2) crashed on TurboQuant reshape
    mismatch. P5b adds `real_page_size_bytes` companion + helper
    resolution (`compute_real_page_size_bytes` /
    `clamp_to_real_shape`) in `kernels/page_size_padded.py` so the
    kernel can consult the natural (un-padded) size even when the
    allocator reserves padded blocks.

    Platform guard: NVIDIA CUDA + SM ≥ 8.0 (shared with TurboQuant).
    """
    name = "P5b KV page-size pad-smaller-to-max (env-opt-in)"
    from vllm._genesis.guards import is_nvidia_cuda, is_amd_rocm, is_cpu_only

    if not is_nvidia_cuda():
        if is_amd_rocm():
            return _skipped(name, "ROCm — TurboQuant not ported")
        if is_cpu_only():
            return _skipped(name, "CPU-only — no TurboQuant KV layer")
        return _skipped(name, "non-NVIDIA platform")

    if not _APPLY_MODE:
        return _applied(name, "dry-run: env-opt-in scaffold ready")

    try:
        from vllm._genesis.wiring.legacy import patch_5b_page_size_pad_smaller
    except Exception as e:
        return _failed(name, f"wiring import failed: {e}")

    status, reason = patch_5b_page_size_pad_smaller.apply()
    if status == "applied":
        return _applied(name, reason)
    if status == "skipped":
        return _skipped(name, reason)
    return _failed(name, reason)


@register_patch("P31 MoE router fp32 softmax")
def apply_patch_31_router_softmax() -> PatchResult:
    """Patch 31: Universal fp32 upcast for MoE router softmax.

    Applies to all GPU vendors — pure-torch primitive. CPU is a no-op in
    practice (no benefit), but doesn't fail.

    Wiring strategy: The callable is made available as
    `vllm._genesis.kernels.router_softmax.router_softmax`. At vLLM engine
    init, the Genesis integration layer (loaded lazily via upstream_compat
    hooks) replaces the upstream `torch.softmax(gating_output, dim=-1)`
    call sites with this function.

    For v7.0-dev, we verify the kernel is importable and report readiness.
    The actual monkey-patch binding happens when vLLM's MoE modules import.
    """
    name = "P31 MoE router fp32 softmax"
    from vllm._genesis.guards import is_cpu_only

    if is_cpu_only():
        return _skipped(
            name,
            "CPU-only platform; fp32 upcast has no numerical benefit here",
        )

    try:
        from vllm._genesis.kernels.router_softmax import router_softmax
        assert callable(router_softmax)
    except Exception as e:
        return _failed(name, f"router_softmax import failed: {e}")

    if not _APPLY_MODE:
        return _applied(name, "dry-run: kernel ready (pass apply=True for live wiring)")

    # Live wiring: wrap grouped_topk router (limited scope — only affects
    # grouped-MoE families; Qwen3.6 uses fused-CUDA-kernel softmax that's
    # out of scope for Python-level rebind).
    try:
        from vllm._genesis.wiring.legacy import patch_31_router_softmax
    except Exception as e:
        return _failed(name, f"wiring import failed: {e}")

    status, reason = patch_31_router_softmax.apply()
    if status == "applied":
        return _applied(name, reason)
    if status == "skipped":
        return _skipped(name, reason)
    return _failed(name, reason)


@register_patch("P22 TurboQuant shared dequant prealloc")
def apply_patch_22_tq_dequant_prealloc() -> PatchResult:
    """Patch 22: Pre-allocate TurboQuant K/V dequant buffers during profile_run.

    Fixes #40420-class OOM at long context: without this patch, dequant buffers
    are allocated lazily inside forward() → invisible to vLLM's memory profiler
    → KV cache over-sized → OOM when a real 234k+ request arrives.

    Platform guard: NVIDIA CUDA + SM ≥ 8.0 (TurboQuant is CUDA-only upstream).

    Wiring strategy: `ensure_turboquant_buffers(impl, layer, device)` is called
    from inside `TurboQuantAttentionImpl._ensure_on_device` via monkey-patch.
    We verify manager is importable and platform-compatible here.
    """
    name = "P22 TurboQuant shared dequant prealloc"
    from vllm._genesis.guards import (
        is_nvidia_cuda, is_sm_at_least, is_amd_rocm, is_cpu_only,
    )
    from vllm._genesis.kernels.dequant_buffer import (
        TurboQuantBufferManager, ensure_turboquant_buffers,
    )

    if not is_nvidia_cuda():
        if is_amd_rocm():
            return _skipped(name, "ROCm — TurboQuant not ported to AMD")
        if is_cpu_only():
            return _skipped(name, "CPU-only — no TurboQuant kernel")
        return _skipped(name, "non-NVIDIA platform")

    if not is_sm_at_least(8, 0):
        return _skipped(name, "SM < 8.0 — TurboQuant requires Ampere+")

    if not TurboQuantBufferManager.should_apply():
        return _skipped(name, "platform guard returned False")

    assert callable(ensure_turboquant_buffers)

    if not _APPLY_MODE:
        return _applied(name, "dry-run: kernel ready (pass apply=True for live wiring)")

    # Live wiring: rebind TurboQuantAttentionImpl._ensure_on_device.
    try:
        from vllm._genesis.wiring.legacy import patch_22_tq_prealloc
    except Exception as e:
        return _failed(name, f"wiring import failed: {e}")

    status, reason = patch_22_tq_prealloc.apply()
    if status == "applied":
        return _applied(name, reason)
    if status == "skipped":
        return _skipped(name, reason)
    return _failed(name, reason)


# PN37 archived 2026-05-04 to vllm/_genesis/_not_used_artifact/.
# Premise (FA2 dead-zone for tiny-Q non-causal) was empirically disproved
# by microbench. Kernel + TDD preserved as research artifact.
# Removed from PATCH_REGISTRY + apply_all so dispatcher matrix doesn't
# show graveyard entries.


@register_patch("P57 TQ spec-decode capture-safe buffers")
def apply_patch_57_spec_decode_capture_safe() -> PatchResult:
    """Patch 57: REAL FIX (proof-of-concept) for vllm-project/vllm#40831.

    Addresses the architectural gap surfaced after deep-diving the
    GDN attention pattern at gdn_attn.py:103-115. TurboQuant declares
    `supports_spec_as_decode=False` AND pre-allocates decode buffers at
    `B=max_num_seqs` shape. Spec-decode batches with q_len=1+num_spec
    cannot fit the captured cudagraph's decode shape — buffer addresses
    captured at warmup don't match runtime addresses → token corruption
    visible as `for for`, `age age`, `<function=call`, etc.

    P57 fixes both layers:
      1. `supports_spec_as_decode = True` based on speculative_config
      2. Buffer alloc B = max_num_seqs * (1 + num_speculative_tokens)

    Status: opt-in via GENESIS_ENABLE_P57_SPEC_DECODE_CAPTURE_SAFE=1.
    Experimental — pending server validation that demonstrates clean
    output WITHOUT cudagraph_mode=NONE workaround. If verified, this
    is a candidate upstream PR.

    Credit: bug surface @noonghunna (vllm#40807, #40831 + six-probe
    ladder noonghunna/qwen36-27b-single-3090@de1d1afa). Reference
    implementation pattern: gdn_attn.py:103-115 by vLLM team.
    """
    name = "P57 TQ spec-decode capture-safe buffers"
    from vllm._genesis.guards import (
        is_nvidia_cuda, is_sm_at_least, is_amd_rocm, is_cpu_only,
    )
    if not is_nvidia_cuda():
        if is_amd_rocm():
            return _skipped(name, "ROCm — TurboQuant not ported")
        if is_cpu_only():
            return _skipped(name, "CPU-only — no TurboQuant kernel")
        return _skipped(name, "non-NVIDIA platform")
    if not is_sm_at_least(8, 0):
        return _skipped(name, "SM < 8.0")
    if not _APPLY_MODE:
        return _applied(name, "dry-run: text-patch ready")
    try:
        from vllm._genesis.wiring.spec_decode import patch_57_spec_decode_capture_safe_buffers
    except Exception as e:
        return _failed(name, f"wiring import failed: {e}")
    status, reason = patch_57_spec_decode_capture_safe_buffers.apply()
    if status == "applied":
        return _applied(name, reason)
    if status == "skipped":
        return _skipped(name, reason)
    return _failed(name, reason)


@register_patch("P56 TQ spec-decode safe-path guard")
def apply_patch_56_spec_decode_guard() -> PatchResult:
    """Patch 56: Workaround for vllm-project/vllm#40831 — TurboQuant ×
    spec-decode degenerate token loops.

    TurboQuant attention backend declares `supports_spec_as_decode=False`
    at `turboquant_attn.py:192` and lacks a varlen kernel analogous to
    FlashAttention's. Spec-decode batches (q_len > 1) get routed through
    a per-row synthetic-decode fast path that breaks GQA causal semantics
    across draft tokens — symptom: degenerate output loops.

    Tightens the fast-path entry condition from
    `q_len <= _CONTINUATION_DECODE_THRESHOLD` to `q_len == 1`, forcing
    spec-decode batches through `_continuation_prefill` (causal-correct
    `flash_attn_varlen_func` path).

    Status: opt-in (`GENESIS_ENABLE_P56_SPEC_DECODE_GUARD=1`).

    Credit: bug surface @noonghunna (vllm-project/vllm#40807, #40831).
    """
    name = "P56 TQ spec-decode safe-path guard"
    from vllm._genesis.guards import (
        is_nvidia_cuda, is_sm_at_least, is_amd_rocm, is_cpu_only,
    )
    if not is_nvidia_cuda():
        if is_amd_rocm():
            return _skipped(name, "ROCm — TurboQuant not ported")
        if is_cpu_only():
            return _skipped(name, "CPU-only — no TurboQuant kernel")
        return _skipped(name, "non-NVIDIA platform")
    if not is_sm_at_least(8, 0):
        return _skipped(name, "SM < 8.0")
    if not _APPLY_MODE:
        return _applied(name, "dry-run: text-patch ready")
    try:
        from vllm._genesis.wiring.spec_decode import patch_56_spec_decode_decode_path_guard
    except Exception as e:
        return _failed(name, f"wiring import failed: {e}")
    status, reason = patch_56_spec_decode_decode_path_guard.apply()
    if status == "applied":
        return _applied(name, reason)
    if status == "skipped":
        return _skipped(name, reason)
    return _failed(name, reason)


@register_patch("P44 TQ mixed-batch attn_out pool")
def apply_patch_44_tq_mixed_attn_out() -> PatchResult:
    """Patch 44: Pool the mixed decode+prefill `attn_out` zeros.

    Complements P26 which pools the prefill-only path. Mixed-batch
    branch (`turboquant_attn.py:438`) previously did
    `torch.zeros(N, Hq, D, dtype=q.dtype)` per forward → up to 80 MB
    zero-init on 4096 token batches. Pool reuses memory + zeroes
    `[:num_tokens]` slice.

    Platform guard: NVIDIA CUDA + SM ≥ 8.0. Default-on.
    """
    name = "P44 TQ mixed-batch attn_out pool"
    from vllm._genesis.guards import (
        is_nvidia_cuda, is_sm_at_least, is_amd_rocm, is_cpu_only,
    )
    if not is_nvidia_cuda():
        if is_amd_rocm():
            return _skipped(name, "ROCm — TurboQuant not ported")
        if is_cpu_only():
            return _skipped(name, "CPU-only — no TurboQuant kernel")
        return _skipped(name, "non-NVIDIA platform")
    if not is_sm_at_least(8, 0):
        return _skipped(name, "SM < 8.0")
    if not _APPLY_MODE:
        return _applied(name, "dry-run: text-patch ready")
    try:
        from vllm._genesis.wiring.legacy import patch_44_tq_mixed_attn_out
    except Exception as e:
        return _failed(name, f"wiring import failed: {e}")
    status, reason = patch_44_tq_mixed_attn_out.apply()
    if status == "applied":
        return _applied(name, reason)
    if status == "skipped":
        return _skipped(name, reason)
    return _failed(name, reason)


@register_patch("P46 GDN gating buffer pool")
def apply_patch_46_gdn_gating_buffers() -> PatchResult:
    """Patch 46: Persistent buffers for `fused_gdn_gating`'s `g` +
    `beta_output` outputs.

    The helper is called once per GDN-bearing layer per forward pass
    and allocates two tiny tensors via `torch.empty(...)`. On
    Qwen3.6-35B-A3B (48 GDN layers) at 250 tok/s decode this is
    ~24 000 allocator ops/sec with zero bytes recovered. Replacing
    with a per-shape-key persistent pool eliminates the churn
    completely (no allocator lock contention, no metadata overhead).

    Byte-exact output vs upstream — Triton kernel writes every
    position unconditionally, so allocated-content doesn't matter
    (equivalent to `torch.empty`).

    Platform guard: NVIDIA CUDA + SM ≥ 8.0. Default-on — no env gate.
    """
    name = "P46 GDN gating buffer pool"
    from vllm._genesis.guards import (
        is_nvidia_cuda, is_sm_at_least, is_amd_rocm, is_cpu_only,
    )

    if not is_nvidia_cuda():
        if is_amd_rocm():
            return _skipped(name, "ROCm — HIP allocator path differs")
        if is_cpu_only():
            return _skipped(name, "CPU-only — no GDN GPU kernel")
        return _skipped(name, "non-NVIDIA platform")

    if not is_sm_at_least(8, 0):
        return _skipped(name, "SM < 8.0 — shares P2x platform gate")

    if not _APPLY_MODE:
        return _applied(name, "dry-run: text-patch ready")

    try:
        from vllm._genesis.wiring.legacy import patch_46_gdn_gating_buffers
    except Exception as e:
        return _failed(name, f"wiring import failed: {e}")

    status, reason = patch_46_gdn_gating_buffers.apply()
    if status == "applied":
        return _applied(name, reason)
    if status == "skipped":
        return _skipped(name, reason)
    return _failed(name, reason)


@register_patch("P7b GDN dual-stream via torch.library.custom_op (opt-in)")
def apply_patch_7b_gdn_dual_stream_customop() -> PatchResult:
    """Patch 7b: graph-safe GDN dual-stream parallelism.

    Alternative to P7 (text-patch with `DualStreamDispatcher` raw CUDA
    streams) that works inside `torch.compile(fullgraph=True)` —
    wraps the two in_proj GEMMs as a single `torch.library.custom_op`
    so dynamo sees an opaque node and doesn't try to trace the stream
    operations.

    Expected gain: +5-8% Qwen3-Next decode tok/s (matches P7 eager
    measurement) while being compatible with vLLM's default
    `aot_compile_fullgraph` path (no `--enforce-eager` required).

    Opt-in via `GENESIS_ENABLE_P7B=1`. Mutually exclusive with P7:
    both text-patch the same 2 lines in `gdn_linear_attn.py`. P7b
    detects P7 conflict via anchor mismatch and skips with a clear
    error.

    Platform guard: NVIDIA CUDA + SM ≥ 8.0.
    """
    name = "P7b GDN dual-stream via torch.library.custom_op (opt-in)"
    from vllm._genesis.guards import (
        is_nvidia_cuda, is_sm_at_least, is_amd_rocm, is_cpu_only,
    )

    if not is_nvidia_cuda():
        if is_amd_rocm():
            return _skipped(name, "ROCm — HIP stream ordering weaker")
        if is_cpu_only():
            return _skipped(name, "CPU-only — no CUDA streams")
        return _skipped(name, "non-NVIDIA platform")

    if not is_sm_at_least(8, 0):
        return _skipped(name, "SM < 8.0 — stream parallelism weak")

    if not _APPLY_MODE:
        return _applied(name, "dry-run: env-opt-in scaffold ready")

    try:
        from vllm._genesis.wiring.legacy import patch_7b_gdn_dual_stream_customop
    except Exception as e:
        return _failed(name, f"wiring import failed: {e}")

    status, reason = patch_7b_gdn_dual_stream_customop.apply()
    if status == "applied":
        return _applied(name, reason)
    if status == "skipped":
        return _skipped(name, reason)
    return _failed(name, reason)


@register_patch("P40 TurboQuant GQA-grouped decode stage1 (opt-in)")
def apply_patch_40_tq_grouped_decode() -> PatchResult:
    """Patch 40: Port upstream PR #40792 GQA-grouped decode stage1 kernel
    for `turboquant_k8v4`.

    Replaces per-head CTA launch (upstream scalar kernel) with
    per-head-group CTA launch (our port). Each CTA handles up to
    BLOCK_H=16 Q heads sharing one KV head → ~4× fewer KV loads,
    2× arithmetic intensity via `tl.dot` on tensor cores.

    Upstream PR body measured +16-27% decode tok/s on Qwen3-32B
    across A100/H100. Our target 2×A5000 (SM 8.6) Qwen3.6-35B-A3B-FP8
    k8v4 should see similar directional gain.

    Opt-in via `GENESIS_ENABLE_P40=1`. Self-retires when upstream PR
    merges (detected by `_tq_grouped_decode_stage1` symbol appearing
    on the upstream module).

    Scope: FP8 keys + 4-bit values only (`turboquant_k8v4`). MSE-key
    presets retain the scalar kernel via dispatcher fallback.

    Platform guard: NVIDIA CUDA + SM ≥ 8.0.
    """
    name = "P40 TurboQuant GQA-grouped decode stage1 (opt-in)"
    from vllm._genesis.guards import (
        is_nvidia_cuda, is_sm_at_least, is_amd_rocm, is_cpu_only,
    )

    if not is_nvidia_cuda():
        if is_amd_rocm():
            return _skipped(name, "ROCm — TurboQuant not ported")
        if is_cpu_only():
            return _skipped(name, "CPU-only — no Triton GPU kernel")
        return _skipped(name, "non-NVIDIA platform")

    if not is_sm_at_least(8, 0):
        return _skipped(name, "SM < 8.0 — Triton tl.dot requires Ampere+")

    if not _APPLY_MODE:
        return _applied(name, "dry-run: rebind ready (pass apply=True for live wiring)")

    try:
        from vllm._genesis.wiring.legacy import patch_40_tq_grouped_decode
    except Exception as e:
        return _failed(name, f"wiring import failed: {e}")

    status, reason = patch_40_tq_grouped_decode.apply()
    if status == "applied":
        return _applied(name, reason)
    if status == "skipped":
        return _skipped(name, reason)
    return _failed(name, reason)


@register_patch("P39a FLA chunk_scaled_dot_kkt persistent A pool")
def apply_patch_39a_fla_kkt_buffer() -> PatchResult:
    """Patch 39a: Persistent `A` buffer for FLA `chunk_scaled_dot_kkt_fwd`.

    GDN chunked-prefill allocates `A = torch.empty(B, T, H, BT, fp32)`
    per-layer per-chunk call. On Qwen3.6-35B-A3B with 32 GDN-bearing
    layers, B=1 T≤4096 H=16 BT=64 fp32 = 16 MiB × 32 = 512 MiB of
    per-step allocator churn during long-context prefill — profiler-
    invisible (lazy inside forward), saturates at the yaml=0.93
    boundary where 12 MiB allocs fail.

    Rewires `chunk_scaled_dot_kkt_fwd` to use a single shared persistent
    pool via `FlaKktBufferManager.acquire`. Pool is sized to max
    `(B, max_num_batched_tokens, H, BT)` at first call; reused across
    all GDN layers (sequential-forward invariant).

    Applied via module-level symbol swap + caller-module rebind (FLA
    typically does `from .chunk_scaled_dot_kkt import
    chunk_scaled_dot_kkt_fwd` → callers capture the original reference;
    we walk `sys.modules` and fix those too).

    Platform guard: NVIDIA CUDA + SM ≥ 8.0 (shared with rest of P2x).

    Expected win: frees the 12-34 MiB runtime-headroom ceiling that was
    blocking yaml ≥ 0.93 on dev134. Enables yaml=0.93-0.94 range that
    the user requested, at chunk=4096.
    """
    name = "P39a FLA chunk_scaled_dot_kkt persistent A pool"
    from vllm._genesis.guards import (
        is_nvidia_cuda, is_sm_at_least, is_amd_rocm, is_cpu_only,
    )

    if not is_nvidia_cuda():
        if is_amd_rocm():
            return _skipped(name, "ROCm — TurboQuant/FLA not ported")
        if is_cpu_only():
            return _skipped(name, "CPU-only — no GDN kernel")
        return _skipped(name, "non-NVIDIA platform")

    if not is_sm_at_least(8, 0):
        return _skipped(name, "SM < 8.0 — FLA GDN requires Ampere+")

    if not _APPLY_MODE:
        return _applied(name, "dry-run: rebind ready (pass apply=True for live wiring)")

    try:
        from vllm._genesis.wiring.legacy import patch_39_fla_kkt_buffer
    except Exception as e:
        return _failed(name, f"wiring import failed: {e}")

    status, reason = patch_39_fla_kkt_buffer.apply()
    if status == "applied":
        return _applied(name, reason)
    if status == "skipped":
        return _skipped(name, reason)
    return _failed(name, reason)


@register_patch("P38 TQ _continuation_prefill persistent workspace")
def apply_patch_38_tq_continuation_memory() -> PatchResult:
    """Patch 38: Replace `_continuation_prefill`'s `.contiguous()` + `torch.cat`
    peak-memory pattern with persistent K_full/V_full shared buffers.

    On dev134+ this path allocates 4× ~128 MiB FP16 transients per call at
    deep prefix continuation (Qwen3.6-35B-A3B-FP8, max_model_len 262144,
    k8v4). Together with allocator fragmentation this saturates a 2×A5000
    setup at cached_len ~= 99k and above — reproducible OOM at
    `turboquant_attn.py:776 v_full = torch.cat(...)`.

    This patch REPLACES the entire `_continuation_prefill` method via
    class-level monkey-patch. The replacement:
      * uses 4-D K/V dequant buffers (prealloc'd by P22's updated helper);
      * writes dequant prefix directly into persistent `_tq_k_full_buf` /
        `_tq_v_full_buf` via in-place `.copy_()` — no `.contiguous()` copy;
      * appends the new chunk into the same workspace instead of
        `torch.cat` → zero transient peaks in the forward path.

    Net budget: +516 MiB persistent (profiler-visible → KV sized correctly)
    to eliminate ~500 MiB of transient-with-fragmentation peaks. This makes
    yaml 0.92-0.94 + chunk 4096 stable for 262k single-request on our 2x
    A5000 setup (previously required yaml=0.80 + chunk=2768 workaround).

    Platform guard: NVIDIA CUDA + SM ≥ 8.0 (shared with P22).
    """
    name = "P38 TQ _continuation_prefill persistent workspace"
    from vllm._genesis.guards import (
        is_nvidia_cuda, is_sm_at_least, is_amd_rocm, is_cpu_only,
    )

    if not is_nvidia_cuda():
        if is_amd_rocm():
            return _skipped(name, "ROCm — TurboQuant not ported")
        if is_cpu_only():
            return _skipped(name, "CPU-only — no TurboQuant kernel")
        return _skipped(name, "non-NVIDIA platform")

    if not is_sm_at_least(8, 0):
        return _skipped(name, "SM < 8.0 — TurboQuant requires Ampere+")

    if not _APPLY_MODE:
        return _applied(name, "dry-run: rebind ready (pass apply=True for live wiring)")

    try:
        from vllm._genesis.wiring.legacy import patch_38_tq_continuation_memory
    except Exception as e:
        return _failed(name, f"wiring import failed: {e}")

    status, reason = patch_38_tq_continuation_memory.apply()
    if status == "applied":
        return _applied(name, reason)
    if status == "skipped":
        return _skipped(name, reason)
    return _failed(name, reason)


@register_patch("P32/P33 TurboQuant cu_2 + synth_seq_lens preallocs")
def apply_patch_32_33_tq_bundled_preallocs() -> PatchResult:
    """Patches 32+33: bundled with P22 — second-hop cu_seqlens scratch (P32)
    and synthetic seq_lens device mirror (P33).

    These are profiler-invisible lazy allocations inside TurboQuant's forward
    path that the master plan identifies as contributing a small but
    real (~0.3% TGS) decode regression when left lazy. We pre-allocate them
    in `_ensure_on_device` alongside the P22 K/V dequant buffers.

    Platform guard: NVIDIA CUDA + SM ≥ 8.0 (shared with P22).

    Wiring: the two get_or_create helpers are called inside
    `ensure_turboquant_buffers()`. This entry-point VERIFIES the helpers
    are importable and platform-compatible and logs the decision.
    """
    name = "P32/P33 TurboQuant cu_2 + synth_seq_lens preallocs"

    try:
        from vllm._genesis.kernels.dequant_buffer import (
            TurboQuantBufferManager,
        )
    except Exception as e:
        return _failed(name, f"kernel import failed: {e}")

    if not TurboQuantBufferManager.should_apply():
        return _skipped(name, "platform guard returned False (shared with P22)")

    # Verify helpers are present (catches migration drift on refactor)
    if not callable(getattr(TurboQuantBufferManager, "get_or_create_cu_2", None)):
        return _failed(name, "get_or_create_cu_2 missing")
    if not callable(
        getattr(TurboQuantBufferManager, "get_or_create_synth_seq_lens", None)
    ):
        return _failed(name, "get_or_create_synth_seq_lens missing")

    return _applied(
        name,
        "cu_2 + synth_seq_lens preallocs registered (invoked from "
        "ensure_turboquant_buffers, fires during profile_run)",
    )


@register_patch("P28 GDN core_attn_out prealloc")
def apply_patch_28_gdn_core_attn() -> PatchResult:
    """Patch 28: Pre-allocate `core_attn_out` in GatedDeltaNet.forward_cuda.

    Previous P19 reverted because the buffer was allocated lazily INSIDE
    forward() (profiler-invisible → CUDA graph recaptures → −30% throughput,
    188× stdev). CRIT-HW-1 from master plan: allocation MUST be via a
    profiler-visible path.

    This correct redo uses `GdnCoreAttnManager.acquire_slice()` which
    reserves the max-size buffer on first call (picked up by profile_run
    warmup) and returns a pointer-stable slice on all subsequent calls.

    Platform guard: NVIDIA CUDA + SM ≥ 8.0. Fallback `torch.zeros` preserves
    correctness on incompatible platforms.

    Wiring strategy: TEXT-PATCH on `gdn_linear_attn.py:571-575`.
    """
    name = "P28 GDN core_attn_out prealloc"
    try:
        from vllm._genesis.kernels.gdn_core_attn_manager import (
            GdnCoreAttnManager,
        )
    except Exception as e:
        return _failed(name, f"manager import failed: {e}")

    # Diagnostic: report whether the platform will actually engage the prealloc.
    engaged = GdnCoreAttnManager.should_apply()

    result = _wiring_text_patch(
        name, "patch_28_gdn_core_attn",
    )
    if result.status == "applied":
        note = "" if engaged else (
            " (applied; runtime will fall back to fresh-zeros on this platform)"
        )
        result = _applied(name, (result.reason or "") + note)
    return result


@register_patch("P7 GDN dual-stream in_proj parallelism")
def apply_patch_7_gdn_dual_stream() -> PatchResult:
    """Patch 7: Parallel execution of `in_proj_qkvz` + `in_proj_ba` GEMMs.

    Recovers ~5% decode throughput on Qwen3-Next / Qwen3.6 hybrid models by
    issuing the two independent GEMMs on separate CUDA streams (aux stream).

    Platform guard:
      - NVIDIA CUDA SM ≥ 8.0: true parallelism (measured +8% on A5000)
      - AMD ROCm:             HIP stream attempt; may serialize
      - Intel XPU / CPU:      sequential fallback (safe)

    Wiring strategy: TEXT-PATCH on `gdn_linear_attn.py` — the two
    back-to-back `in_proj_*` calls in forward_cuda are replaced with a
    `DualStreamDispatcher.maybe_parallel(...)` call that chooses parallel
    or sequential execution based on platform.
    """
    name = "P7 GDN dual-stream in_proj parallelism"
    from vllm._genesis.guards import is_cpu_only, is_intel_xpu
    from vllm._genesis.kernels.gdn_dual_stream import DualStreamDispatcher

    # Always initialize the dispatcher (diagnostics) even in dry-run mode.
    parallel_ok = DualStreamDispatcher.init_once()
    if parallel_ok:
        log.info("[Genesis P7] dispatcher ready (parallel path)")
    else:
        log.info("[Genesis P7] dispatcher ready (sequential fallback)")

    if is_cpu_only():
        # Still register wiring in apply mode so a GPU worker spawned from
        # the same install tree sees the patch. But note the zero-benefit.
        note = " — CPU has no stream parallelism, functional fallback only"
    elif is_intel_xpu():
        note = " — XPU falls back to sequential"
    else:
        note = ""

    result = _wiring_text_patch(
        name, "patch_7_gdn_dual_stream",
    )
    if result.status == "applied" and note:
        result = _applied(name, (result.reason or "") + note)
    return result


@register_patch("P17/P18 Marlin MoE per-SM tuning")
def apply_patch_17_18_marlin_tuning() -> PatchResult:
    """Patches 17+18: Per-SM optimal Marlin MoE `block_size_m` selection.

    Upstream heuristic lands on bsm=16 for FP8. On A5000 (SM 8.6) + Qwen3.6
    M≤4, topk=8, E=256, bsm=8 is measured +1.2%. Additional env knobs allow
    manual tuning of num_warps and num_stages.

    Platform guard: NVIDIA CUDA only (Marlin is a CUDA kernel).

    Wiring strategy: `get_optimal_block_size_m()` is consulted by vLLM's
    fused_marlin_moe dispatcher via monkey-patch. Env overrides:
      VLLM_MARLIN_MOE_BLOCK_SIZE_M  → bsm override (8/16/32/48/64)
      VLLM_MARLIN_MOE_NUM_WARPS     → warp count (2/4/8)
      VLLM_MARLIN_MOE_NUM_STAGES    → pipeline stages (1-8)
    """
    name = "P17/P18 Marlin MoE per-SM tuning"
    from vllm._genesis.guards import is_nvidia_cuda, get_compute_capability
    from vllm._genesis.kernels.marlin_tuning import (
        get_optimal_block_size_m,
        get_num_warps_override,
        get_num_stages_override,
    )

    if not is_nvidia_cuda():
        return _skipped(name, "non-NVIDIA — Marlin is CUDA-only")

    cc = get_compute_capability()
    bsm = get_optimal_block_size_m()
    warps = get_num_warps_override()
    stages = get_num_stages_override()

    if bsm is None:
        return _skipped(
            name,
            f"no tuning entry for SM {cc} — upstream heuristic will be used",
        )

    log.info(
        "[Genesis P17/P18] Marlin tuning ready: SM=%s bsm=%d "
        "num_warps=%s num_stages=%s",
        cc, bsm,
        warps if warps is not None else "default",
        stages if stages is not None else "default",
    )
    return _applied(name, f"SM={cc} bsm={bsm}")


@register_patch("P14 block_table tail zero-fill")
def apply_patch_14_block_table_tail_zero() -> PatchResult:
    """Patch 14: Zero the tail of block_table row after append/move.

    Fixes silent divergence from stale block IDs leaking past
    `num_blocks_per_row` when a block_table row slot is reused by a shorter
    request after a longer one (vLLM PR #39591 / issue #39589).

    Platform guard: universal (pure numpy/torch indexing — no vendor deps).

    Wiring strategy (v7.0 step 5): runtime class-method monkey-patch on
    `vllm.v1.worker.block_table.BlockTable.append_row` and `move_row`.
    Wrapped versions call the original then tail-zero with our helper.
    """
    name = "P14 block_table tail zero-fill"

    try:
        from vllm._genesis.kernels.block_table_zero import zero_block_table_tail
        assert callable(zero_block_table_tail)
    except Exception as e:
        return _failed(name, f"kernel import failed: {e}")

    if not _APPLY_MODE:
        return _applied(name, "dry-run: kernel ready (pass apply=True for live wiring)")

    try:
        from vllm._genesis.wiring.legacy import patch_14_block_table
    except Exception as e:
        return _failed(name, f"wiring import failed: {e}")

    status, reason = patch_14_block_table.apply()
    if status == "applied":
        return _applied(name, reason)
    if status == "skipped":
        return _skipped(name, reason)
    return _failed(name, reason)


@register_patch("P18b TurboQuant decode stage1 tune")
def apply_patch_18b_tq_decode_tune() -> PatchResult:
    """Patch 18b: Env-driven TurboQuant decode stage1 kernel tunables.

    Exposes BLOCK_KV / num_warps / num_stages via env vars so non-H100 cards
    (A5000 especially) can re-tune away from H100-shaped defaults.

    Platform guard: NVIDIA CUDA + SM 8.0+ (TurboQuant is CUDA-only).

    Wiring strategy: `resolve_decode_tune()` is consulted by the kernel
    launcher in `triton_turboquant_decode.py` via monkey-patch or text-
    replacement (Triton compile-time params can't be monkey-patched; text
    patcher for those literals).
    """
    name = "P18b TurboQuant decode stage1 tune"
    from vllm._genesis.kernels import tq_decode_tune as t

    if not t.should_apply():
        return _skipped(
            name,
            "non-NVIDIA or pre-Ampere — TurboQuant not applicable",
        )

    # Log and report whether user opted into overrides
    t.log_selected_tune()

    if t.has_any_override():
        bkv, nw, ns = t.resolve_decode_tune()
        return _applied(name, f"env override BLOCK_KV={bkv} warps={nw} stages={ns}")

    return _applied(
        name,
        f"no env override — using upstream defaults "
        f"({t.UPSTREAM_BLOCK_KV}/{t.UPSTREAM_NUM_WARPS}/{t.UPSTREAM_NUM_STAGES})",
    )


@register_patch("P20 TurboQuant continuation-prefill FP16 rotate")
def apply_patch_20_tq_continuation_prefill() -> PatchResult:
    """Patch 20: Halve peak memory of `_continuation_prefill` (fixes #40420).

    Replaces upstream's FP32 rotation + redundant `.contiguous()` with a
    single FP16 matmul + non-contiguous view that torch.cat materializes.

    Platform guard: NVIDIA CUDA + SM 8.0+ (TurboQuant is CUDA-only).

    Wiring strategy: `continuation_prefill_fp16_rotate()` replaces the
    4-step fp32 block in `TurboQuantAttentionImpl._continuation_prefill`
    via monkey-patch.
    """
    name = "P20 TurboQuant continuation-prefill FP16 rotate"
    from vllm._genesis.kernels import tq_continuation_prefill as t

    if not t.should_apply():
        return _skipped(
            name,
            "non-NVIDIA or pre-Ampere — TurboQuant not applicable",
        )

    # Verify helpers importable
    try:
        assert callable(t.continuation_prefill_fp16_rotate)
        assert callable(t.continuation_prefill_k_view_fp8)
        assert callable(t.continuation_prefill_v_view)
        assert callable(t.get_pi_half)
    except Exception as e:
        return _failed(name, f"helper import failed: {e}")

    log.info(
        "[Genesis P20] TQ _continuation_prefill FP16 helpers ready for "
        "TurboQuantAttentionImpl hook"
    )
    return _applied(name, "fp16-rotation helper ready for _continuation_prefill hook")


@register_patch("P1/P2 FP8 kernel dispatcher")
def apply_patch_1_2_fp8_dispatcher() -> PatchResult:
    """Patches 1+2: FP8 kernel path selection (Triton native vs Marlin fallback).

    Upstream `TritonBlockFP8ScaledMMKernel` assumes SM ≥ 8.9. On Ampere
    (SM 8.0/8.6), it silently produces wrong numerics. This dispatcher routes
    Ampere to Marlin fallback and Ada/Hopper/Blackwell to native Triton.

    Platform guard: NVIDIA CUDA only.

    Wiring strategy: `should_skip_triton_fp8()` is consulted by vLLM's FP8
    kernel dispatcher via monkey-patch on `TritonBlockFP8ScaledMMKernel`.
    """
    name = "P1/P2 FP8 kernel dispatcher"
    from vllm._genesis.guards import is_nvidia_cuda, get_compute_capability
    from vllm._genesis.kernels.fp8_dispatcher import (
        requires_marlin_fp8_fallback,
        fp8_triton_kernel_supported,
        log_dispatcher_decision,
    )

    if not is_nvidia_cuda():
        return _skipped(name, "non-NVIDIA — different FP8 path")

    cc = get_compute_capability()
    log_dispatcher_decision()

    if requires_marlin_fp8_fallback():
        return _applied(name, f"SM={cc} → Marlin fallback path selected")

    if fp8_triton_kernel_supported():
        return _applied(name, f"SM={cc} → native Triton FP8 path selected")

    return _skipped(
        name, f"SM={cc} — no FP8 support at all (unexpected on NVIDIA)",
    )


# ═══════════════════════════════════════════════════════════════════════════
#                             MAIN ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════

def run(verbose: bool = True, apply: bool = False) -> PatchStats:
    """Apply all registered patches, return statistics.

    Args:
        verbose: If True, log platform summary before applying patches.
        apply:   If True, perform the actual wiring (text-patches on disk +
                 runtime attribute rebinds). If False (default), run in
                 DRY-RUN mode: import kernels, verify platform compat, but
                 do NOT rewrite any files or rebind any attributes. Dry-run
                 is the right default because it's safe from anywhere.

                 apply=True should be passed from:
                   - The vLLM plugin register() entry point (once per process)
                   - The container entrypoint script (for text-patches that
                     must land before `vllm serve` starts)

    Returns:
        PatchStats with counts and details per patch.
    """
    # Configure logging if not already configured
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="[%(levelname)s:%(name)s] %(message)s",
        )

    # Propagate apply mode to patch functions via module-level flag.
    global _APPLY_MODE
    _APPLY_MODE = apply

    # [Genesis T4.6] Compile-time watchdog — log total apply elapsed.
    # Triton kernel pre-build (e.g. PR41422 _build_kernel() at apply()) can
    # take 30-90s on cold cache. >120s is a red flag (autotune regression
    # or stale cache mismatch) — investigate before user requests start.
    import time
    _t0_apply = time.perf_counter()

    stats = PatchStats()

    # Platform diagnostic — helps debugging on unexpected hardware
    try:
        from vllm._genesis.guards import platform_summary
        summary = platform_summary()
        if verbose:
            log.info("Genesis platform: %s",
                     json.dumps(summary, default=str, indent=None))
    except Exception as e:
        log.warning("Platform summary failed: %s", e)

    # [Genesis pin-gate] Sander 2026-05-04 — "защита от дурака". Runs in
    # BOTH plugin auto-load (run() called from register()) AND CLI PRE-pass
    # (run() called from main()). Strict mode = sys.exit(2) on unknown pin.
    try:
        from vllm._genesis.guards import (
            assert_vllm_pin_allowed,
            get_vllm_full_version_string,
            KNOWN_GOOD_VLLM_PINS,
        )
        pin = get_vllm_full_version_string() or "unknown"
        log.info("[Genesis pin-gate] running vllm pin = %s", pin)
        log.info(
            "[Genesis pin-gate] allowlist (%d entries): %s",
            len(KNOWN_GOOD_VLLM_PINS), list(KNOWN_GOOD_VLLM_PINS),
        )
        status, message = assert_vllm_pin_allowed()
        if status == "ok":
            log.info("[Genesis pin-gate] OK — %s", message)
        else:
            log.warning("[Genesis pin-gate] %s — %s", status.upper(), message)
    except SystemExit:
        # strict-mode hard-stop already printed; propagate exit
        raise
    except Exception as e:
        log.warning("[Genesis pin-gate] check skipped (error: %s)", e)

    # PDL misconfig check (vLLM issue #40742). Warn loudly but don't fail —
    # some environments set these globally and other GPUs in the cluster use
    # them. On the local Ampere rank, we just advise unsetting.
    try:
        from vllm._genesis.guards import detect_pdl_env_misconfig
        bad = detect_pdl_env_misconfig()
        if bad:
            log.warning(
                "[Genesis guard] PDL env vars set but this GPU does NOT "
                "support PDL safely: %s. Reference: vLLM issue #40742 "
                "(Inductor autotune + torch.cuda.synchronize() inside CUDA "
                "graph capture → illegal cuda op → engine crash). Consider "
                "unsetting these on this node.",
                bad,
            )
    except Exception as e:
        log.debug("PDL misconfig check failed: %s", e)

    # Banner
    log.info(
        "Genesis Unified Patch v7.0 — Ampere FP8 + TQ + MoE + Hybrid + bugfixes. "
        "Philosophy: МЫ ЧИНИМ, НЕ ЛОМАЕМ."
    )

    # Validate PATCH_REGISTRY shape + dependency graph at boot. Issues are
    # logged so operators see drift (e.g. unknown env_flag pattern, missing
    # superseded_by on deprecated patch, requires_patches referencing an
    # unknown ID). ERROR-level issues are surfaced loudly; WARNING are
    # logged at INFO so they don't drown the boot log on a busy registry.
    # The registry IS the contract — silent drift is the failure mode this
    # block was added to catch.
    try:
        from vllm._genesis.dispatcher import (
            PATCH_REGISTRY as _GENESIS_DISPATCHER_REGISTRY,
            validate_registry,
        )
        registry_issues = validate_registry()
        for i in registry_issues:
            if i.severity == "ERROR":
                log.error(
                    "[Genesis registry] %s: %s",
                    i.patch_id, i.message,
                )
            elif i.severity == "WARNING":
                log.warning(
                    "[Genesis registry] %s: %s",
                    i.patch_id, i.message,
                )
            else:
                log.info(
                    "[Genesis registry] %s: %s",
                    i.patch_id, i.message,
                )
        if verbose:
            n_err = sum(1 for i in registry_issues if i.severity == "ERROR")
            if n_err == 0:
                log.info(
                    "[Genesis registry] %d dispatcher entries — "
                    "schema-clean, dependency graph consistent.",
                    len(_GENESIS_DISPATCHER_REGISTRY),
                )
            else:
                log.error(
                    "[Genesis registry] %d entries — %d ERROR(s) above. "
                    "Apply will continue but operators must investigate.",
                    len(_GENESIS_DISPATCHER_REGISTRY), n_err,
                )
    except Exception as e:
        log.debug("[Genesis registry] validation skipped: %s", e)

    # GPU profile + per-patch recommendations (suggest-only, never auto-enables)
    try:
        from vllm._genesis.gpu_profile import print_recommendations
        rec_text = print_recommendations(stream=None)
        for line in rec_text.split("\n"):
            log.info(line)
    except Exception as e:
        log.debug("[gpu_profile] recommendation skipped: %s", e)

    # [Phase 5b plugins] Discover + register community plugin patches
    # via setuptools entry-points. OPT-IN: only fires when
    # GENESIS_ALLOW_PLUGINS=1. Default behavior: zero foreign code loaded.
    try:
        from vllm._genesis.compat.plugins import (
            register_plugins as _register_genesis_plugins,
        )
        n_plugins = _register_genesis_plugins()
        if n_plugins > 0:
            log.info(
                "[Genesis plugins] registered %d community patch(es) via "
                "entry-points (lifecycle=community).", n_plugins,
            )
    except Exception as e:
        log.debug("[plugins] discovery skipped: %s", e)

    # G-006 fix (audit 2026-05-02): Phase 5c apply_callable plugin pass
    # was previously HERE (BEFORE core patch loop), contradicting the
    # docstring "After core patches finish, walk plugins". Moved BELOW
    # the core patch loop (just before telemetry) so plugin authors can
    # rely on core patches being already applied — they may text-patch
    # files that core patches have already modified, and need to find
    # the post-modification anchors.

    # [Phase 5d telemetry] Opt-in anonymized telemetry. Default OFF —
    # only fires when GENESIS_ENABLE_TELEMETRY=1. Even when ON, only
    # saves locally. Network upload is a separate gate
    # (GENESIS_TELEMETRY_UPLOAD=1) and is currently a no-op until the
    # community dashboard is live.
    try:
        from vllm._genesis.compat.telemetry import (
            is_enabled as _telemetry_is_enabled,
            collect_report as _telemetry_collect_report,
            save_report as _telemetry_save_report,
        )
        if _telemetry_is_enabled():
            report = _telemetry_collect_report()
            path = _telemetry_save_report(report)
            if path:
                log.info(
                    "[Genesis telemetry] anonymized report saved → %s "
                    "(no network upload — see telemetry CLI)", path,
                )
    except Exception as e:
        log.debug("[telemetry] save skipped: %s", e)

    # Apply each patch
    for patch_name, patch_fn in PATCH_REGISTRY:
        try:
            result = patch_fn()
            if not isinstance(result, PatchResult):
                # Back-compat: legacy bool return
                result = (
                    _applied(patch_name) if result
                    else _failed(patch_name, "patch_fn returned False")
                )
            stats.results.append(result)
            if result.status == "failed":
                log.error("[Genesis] FAILED: %s — %s",
                          result.name, result.reason)
            elif result.status == "skipped":
                # 2026-04-28: anchor drift / required_anchor_missing is a
                # latent risk (patch silently not protecting). Surface as
                # WARNING so operators notice in boot logs. Other skip
                # reasons (opt-in, deprecated, redundant) stay at INFO.
                _is_drift = (
                    "required anchor" in result.reason.lower()
                    or "required_anchor_missing" in result.reason.lower()
                    or "anchor not found" in result.reason.lower()
                    or "ambiguous_anchor" in result.reason.lower()
                )
                if _is_drift:
                    log.warning("[Genesis] DRIFT skipped: %s — %s",
                                result.name, result.reason)
                else:
                    log.info("[Genesis] skipped: %s — %s",
                             result.name, result.reason)
            else:
                log.info("[Genesis] applied: %s — %s",
                         result.name, result.reason)
        except Exception as e:
            stats.results.append(
                _failed(patch_name, f"{type(e).__name__}: {e}")
            )
            log.exception("[Genesis] EXCEPTION in %s", patch_name)

    log.info("Genesis %s", stats)

    # [Genesis v7.65 / Cliff 8 hardening] Surface partial-apply warnings.
    # Silent anchor-drift / ambiguous-anchor / anchor-missing skips were
    # the class noonghunna flagged in club-3090 discussion #19. Drift
    # detection works correctly, but the user-visible summary previously
    # buried the signal in the same `skipped` count as opt-in OFF. Now
    # warnings are pulled out and logged individually at WARNING level.
    if stats.partial_apply_warnings:
        log.warning(
            "[Genesis] %d partial-apply warning(s) — patch(es) failed to "
            "match expected source pattern. Review below to confirm anchor "
            "drift vs upstream change vs config issue:",
            stats.partial_apply_warnings_count,
        )
        for r in stats.partial_apply_warnings:
            log.warning("[Genesis] ⚠️  %s — %s", r.name, r.reason)

    # [Genesis v7.13] Emit Dispatcher v2 apply matrix as a single readable
    # block. Only matters for patches that route through dispatcher.should_apply
    # (P56-PR36138 currently); other patches get only the per-line INFO above.
    try:
        from vllm._genesis.dispatcher import log_apply_matrix
        log_apply_matrix()
    except Exception as e:
        log.debug("[Genesis] dispatcher matrix dump failed (non-fatal): %s", e)

    # [Genesis A3/D2] Validate dependencies / conflicts on the actual
    # APPLY set. Static registry validation runs first (cheap, catches
    # typos in requires_patches/conflicts_with refs), then runtime plan
    # check. Issues are logged at ERROR/WARNING level — we do NOT abort
    # boot here because operators may have legitimate reasons for unusual
    # combinations during diagnosis.
    try:
        from vllm._genesis.dispatcher import (
            validate_registry, validate_apply_plan,
            log_validation_issues, get_apply_matrix,
        )
        static_issues = validate_registry()
        if static_issues:
            log_validation_issues(static_issues)
        applied_set = {d["patch_id"] for d in get_apply_matrix() if d["applied"]}
        plan_issues = validate_apply_plan(applied_set)
        log_validation_issues(plan_issues)
    except Exception as e:
        log.debug("[Genesis] dispatcher validator unavailable: %s", e)

    # [Phase 5c apply_callable, G-006 audit fix 2026-05-02] After the
    # core patch loop finishes, walk plugins whose env flags are set
    # and call their apply_callable. Plugin failures are isolated
    # (logged, counted, never crash apply_all). Skipped when
    # GENESIS_ALLOW_PLUGINS gate is closed. Re-runs validate_registry
    # so plugin entries injected at register_plugins() time are
    # included in the boot-time validation pass (G-007 fix).
    if apply:
        try:
            from vllm._genesis.compat.plugins import apply_all_plugins
            plugin_stats = apply_all_plugins()
            if plugin_stats.get("total", 0) > 0:
                log.info(
                    "[Genesis plugins] apply pass: total=%d applied=%d "
                    "skipped=%d failed=%d",
                    plugin_stats["total"], plugin_stats["applied"],
                    plugin_stats["skipped"], plugin_stats["failed"],
                )
                # G-007 fix: re-validate registry now that plugin entries
                # were potentially added during register_plugins().
                try:
                    from vllm._genesis.dispatcher import validate_registry
                    post_plugin_issues = validate_registry()
                    n_plugin_err = sum(
                        1 for i in post_plugin_issues if i.severity == "ERROR"
                    )
                    if n_plugin_err > 0:
                        log.error(
                            "[Genesis registry] post-plugin validation: "
                            "%d ERROR(s) — operator should investigate",
                            n_plugin_err,
                        )
                        for i in post_plugin_issues:
                            if i.severity == "ERROR":
                                log.error(
                                    "[Genesis registry plugin] %s: %s",
                                    i.patch_id, i.message,
                                )
                except Exception as ve:
                    log.debug(
                        "[Genesis registry] post-plugin validation skipped: %s",
                        ve,
                    )
        except Exception as e:
            log.debug("[plugins] apply pass skipped: %s", e)

    # [Genesis T4.6] Compile-time watchdog post-summary.
    _elapsed = time.perf_counter() - _t0_apply
    if _elapsed > 120:
        log.warning(
            "[Genesis compile-watchdog] apply_all took %.1fs (>120s threshold) — "
            "investigate Triton compile cache state, autotune regression, or "
            "stale .so files. Consider clearing TRITON_CACHE_DIR + retrying.",
            _elapsed,
        )
    elif _elapsed > 60:
        log.info(
            "[Genesis compile-watchdog] apply_all elapsed: %.1fs (warm cache "
            "should be < 30s; first cold-compile boot may take up to 90s)",
            _elapsed,
        )
    else:
        log.info("[Genesis compile-watchdog] apply_all elapsed: %.1fs", _elapsed)
    stats.compile_elapsed_sec = _elapsed

    # ─────────────────────────────────────────────────────────────────
    # [v7.72.2 fix 2026-05-05] Structured boot summary emit point.
    #
    # MUST live in run() (not main()) — vllm's plugin loader calls run()
    # via the genesis_v7 entry point, never main(). Putting the summary
    # only in main() meant it appeared on `python3 -m vllm._genesis.
    # patches.apply_all` CLI runs but NEVER on real production boot.
    # This regression silently shipped between v7.70 and v7.72.2.
    #
    # Falls back to v7.13 apply matrix on any error so boot keeps
    # working. Errors logged at WARN so operators see them (not the old
    # silent debug log that hid the bug).
    # ─────────────────────────────────────────────────────────────────
    try:
        from vllm._genesis.dispatcher import log_structured_boot_summary
        log_structured_boot_summary()
    except Exception as e:
        log.warning(
            "[Genesis] structured boot summary unavailable (%s: %s) — "
            "falling back to v7.13 apply matrix. Check "
            "dispatcher.dump_structured_boot_summary().",
            type(e).__name__, e,
        )
        try:
            from vllm._genesis.dispatcher import log_apply_matrix
            log_apply_matrix()
        except Exception as e2:
            log.warning(
                "[Genesis] v7.13 apply matrix fallback also unavailable: %s: %s",
                type(e2).__name__, e2,
            )

    return stats


def verify_live_rebinds() -> dict[str, Any]:
    """Post-register verification: confirm runtime rebinds are actually live
    in the current process (TDD discipline from master plan Part 3).

    Returns a dict:
      {
        "P22": {"expected": True, "actual": True, "ok": True},
        "P31": {"expected": True, "actual": True, "ok": True},
        "P14": {"expected": True, "actual": True, "ok": True},
        ...
      }

    Only patches with Python-attribute rebinds are checked. Text-patches
    (P3, P4, P5, P6, P8, P15) modify source files and are verified by the
    diagnostic probes in validate_integration.sh (grep file for markers).

    Usage (end-of-register hook or test):
      from vllm._genesis.patches.apply_all import verify_live_rebinds
      results = verify_live_rebinds()
      for name, r in results.items():
          if not r["ok"]:
              log.warning("[Genesis] rebind %s not live: expected=%s actual=%s",
                          name, r["expected"], r["actual"])
    """
    results: dict[str, dict] = {}

    def _check(patch_id: str, wiring_module: str):
        """Invoke `is_applied()` on the wiring module; record result."""
        try:
            import importlib
            mod = importlib.import_module(
                _resolve_wiring_module(wiring_module)
            )
        except Exception as e:
            results[patch_id] = {
                "expected": True, "actual": False, "ok": False,
                "error": f"import failed: {e}",
            }
            return
        is_applied_fn = getattr(mod, "is_applied", None)
        if is_applied_fn is None or not callable(is_applied_fn):
            results[patch_id] = {
                "expected": True, "actual": None, "ok": True,
                "note": "module has no is_applied() — skipped",
            }
            return
        try:
            actual = bool(is_applied_fn())
        except Exception as e:
            results[patch_id] = {
                "expected": True, "actual": False, "ok": False,
                "error": f"is_applied() raised: {e}",
            }
            return
        results[patch_id] = {
            "expected": True, "actual": actual, "ok": actual,
        }

    # Runtime rebinds (set attrs on live vLLM classes/modules)
    _check("P22", "patch_22_tq_prealloc")
    _check("P31", "patch_31_router_softmax")
    _check("P14", "patch_14_block_table")
    _check("P28", "patch_28_gdn_core_attn")
    # v7.2 / v7.3 additions — both have symmetric `apply/is_applied/revert`
    # trios per patch_38/patch_39 wiring surface contracts.
    _check("P38", "patch_38_tq_continuation_memory")
    _check("P39a", "patch_39_fla_kkt_buffer")

    return results


def main() -> int:
    """CLI entrypoint. Returns exit code.

    CLI default is apply=True because this entrypoint is the one invoked
    from container scripts (pre-vllm-serve) where text-patches MUST land.
    Pass `--dry-run` for diagnosis-only mode.
    Pass `--verify-rebinds` for post-register verification (additional
    verification + non-zero exit code if any rebind not live).

    Per Sander 2026-05-04: enforce vllm pin allowlist (защита от дурака).
    Set GENESIS_VLLM_PIN_POLICY=strict in production start scripts to
    sys.exit(2) on unknown pin instead of just warning.
    """
    import sys as _sys
    argv = _sys.argv[1:]
    dry = "--dry-run" in argv
    verify = "--verify-rebinds" in argv

    # Pin allowlist gate is now in run() so it triggers on every entry path
    # (CLI + plugin auto-load). No need to duplicate it here.

    try:
        stats = run(verbose=True, apply=not dry)
    except Exception as e:
        log.exception("Genesis orchestrator setup error: %s", e)
        return 2

    # NOTE: structured boot summary already emitted by run() above.
    # (v7.72.2 fix moved the call from main() into run() so the plugin
    # entry point — which only invokes run() — also gets the summary.)

    exit_code = 1 if stats.failed_count > 0 else 0

    if verify:
        log.info("[Genesis] Post-register rebind verification:")
        results = verify_live_rebinds()
        any_failed = False
        for patch_id, r in results.items():
            mark = "✓" if r.get("ok") else "✗"
            extra = r.get("error") or r.get("note") or ""
            log.info(
                "  %s %s expected=%s actual=%s %s",
                mark, patch_id, r.get("expected"), r.get("actual"), extra,
            )
            if not r.get("ok"):
                any_failed = True
        if any_failed:
            exit_code = max(exit_code, 1)

    return exit_code


def set_apply_mode(mode: bool) -> None:
    """Set dry-run (False) vs apply (True) mode out-of-band.

    Used by `compat.verify` B1 dry-run check before calling `run_apply_all()`.
    """
    global _APPLY_MODE
    _APPLY_MODE = mode


def run_apply_all(verbose: bool = True, apply: bool | None = None) -> PatchStats:
    """Back-compat alias for :func:`run` (consumed by `compat.verify`).

    When ``apply`` is None the current module ``_APPLY_MODE`` (set via
    :func:`set_apply_mode`) is honored.
    """
    return run(verbose=verbose, apply=_APPLY_MODE if apply is None else apply)


# ── Single-registry derived view ────────────────────────────────────────────
# Built once, at import end, after every `@register_patch` has attached its
# callable + order onto `dispatcher.PATCH_REGISTRY`. Read-only ordered
# projection — NOT a second registry. `run()` iterates it; a few legacy tests
# import it as `[(name, fn), ...]`. Metadata-only entries (no `apply_callable`)
# are excluded by design.
def _build_apply_view() -> list[tuple[str, Callable[[], PatchResult]]]:
    executable = [
        (m.get("_apply_order", 1_000_000),
         m.get("_display_name", pid),
         m["apply_callable"])
        for pid, m in _META_REGISTRY.items()
        if callable(m.get("apply_callable"))
    ]
    executable.sort(key=lambda t: t[0])
    return [(name, fn) for _, name, fn in executable]


# Bind the 85 metadata-driven (text-patch / rebind) patches onto their registry
# entries before projecting the view. Outliers above already bound themselves
# via @register_patch.
_bind_wiring_patches()

PATCH_REGISTRY: list[tuple[str, Callable[[], PatchResult]]] = _build_apply_view()


if __name__ == "__main__":
    sys.exit(main())
