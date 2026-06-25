# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Genesis is a **runtime patcher for vLLM** — *not* a fork, quantizer, or inference engine. It
pins to a specific vLLM commit and applies ~110 small, surgical changes at boot (text-edits at
known upstream source anchors, class-method rebinds, middleware installs, plus Genesis-original
Triton kernels) that turn stock vLLM into a production Qwen3.6 server on consumer NVIDIA GPUs
(3090 / A5000 / 4090 / 5090 / H20). Patches **auto-retire** when upstream merges the underlying
fix, and are pruned when their upstream PR is closed-unmerged.

The package lives at `vllm/_genesis/` — a PEP 420 namespace sub-package mounted into vLLM's
`site-packages` so patches can run before the engine imports the files they edit.

**Fork note:** this checkout is a maintained fork; the original `genesis-vllm-patches` appears
abandoned. README badges/docs still link to the upstream `Sandermage/genesis-vllm-patches` repo
and may be stale — treat in-tree code/tests as the source of truth over doc cross-links.

## Core architecture

The patch pipeline, boot to apply:

1. **Plugin entry point** — `tools/genesis_vllm_plugin/genesis_v7/__init__.py::register()` is
   registered under vLLM's `vllm.general_plugins` entry point, so vLLM calls it once per
   engine/rank process at startup. It must be idempotent and **never raise** (log + return on
   error — never block engine boot). It calls `apply_all.run(apply=True)`.
2. **Orchestrator** — `vllm/_genesis/patches/apply_all.py` attaches each patch's apply step onto its
   `dispatcher` metadata entry (as `apply_callable` + `_display_name` + `_apply_order`), then builds
   `apply_all.PATCH_REGISTRY` — a *derived*, ordered `[(name, fn), …]` view of the single registry,
   not an independent store. Two binding paths: the **generic metadata-driven executor**
   (`_apply_wiring_entry`, bound by `_bind_wiring_patches` from each entry's `wiring` field — the ~84
   text-patch/rebind patches) and **hand-written `@register_patch apply_patch_*` functions** (the ~22
   outliers with real apply-time logic). Boot order is the explicit `_APPLY_ORDER` list. Each apply
   step returns a `PatchResult` (`applied`/`skipped`/`failed`); `run()` iterates the view and prints
   the structured boot summary.
3. **Wiring** — each patch's real edits live in a `wiring/<category>/patch_<id>_*.py` module whose
   `apply()` returns `(status, reason)`. A registry entry names its module via the `wiring: "<stem>"`
   field (the executor imports + runs it); outlier functions import it directly. Wiring uses the
   `TextPatcher` / `TextPatch` framework in `wiring/text_patch.py` (plus `MultiFilePatchTransaction`
   for atomic multi-file edits with rollback).
4. **The single registry** — `vllm/_genesis/dispatcher.py::PATCH_REGISTRY` (a dict keyed by patch
   ID) is the **sole** source of truth for both metadata (`title`, `env_flag`, `env_flag_aliases`,
   `default_on`, `category`, `credit`, `upstream_pr`, `applies_to`, `conflicts_with`,
   `requires_patches`, and `wiring` — the module stem that implements the patch) and the apply
   callable (attached at import by `_bind_wiring_patches` for `wiring`-declared patches, or
   `@register_patch` for outliers).
   `should_apply(id)` combines it with `model_detect` / `config_detect` / env flags.
   `python3 -m vllm._genesis.dispatcher` dumps the full decision matrix.
5. **Guards** — `vllm/_genesis/guards.py` is the *only* place vendor/chip/model/dep detection
   lives (`is_nvidia_cuda`, `is_sm_at_least`, `is_model_arch`, …). Fail-safe: returns a safe
   default on any exception, and snapshots platform facts at module-load time so `torch.dynamo`
   can trace through the guards.

Supporting layers: `compat/` (the `genesis` CLI + diagnostics — `doctor`, `explain`, `verify`,
`preflight`, model registry, lifecycle audit, schema validator), `kernels/` (Genesis-original
Triton kernels), `middleware/` (request-level pre-engine logic).

### Single registry (collapsed 2026-06)

There used to be *two* registries (an `apply_all` list + the `dispatcher` dict) kept in sync by
`test_apply_all_dispatcher_sync.py`. They were collapsed: `dispatcher.PATCH_REGISTRY` is now the
only store, `@register_patch` writes the callable into it, and that sync test is gone (consistency
is structural — a callable can't exist without an entry). **113** entries today; **106** are
executable (carry an `apply_callable`), **7** are metadata-only diagnostics with no apply function
(`P51`, `P69`, `P102`, `PN60`, `PN63`, `PN64`, `PN40-classifier`). `test_dispatcher_validator.py`
still validates entry shape + dependency refs. (Exact counts drift as patches are added/pruned —
`python3 -m vllm._genesis.dispatcher` and the `docs/PATCHES.md` header are the live source; the
latter's total is gated by `test_patches_md_sync.py`.)

A second collapse (2026-06) removed the per-patch *function* ceremony. Of the **106** executable
entries, **84** are now *metadata-driven*: their apply step is a pure text-patch/rebind dispatch
declared by a `wiring: "<stem>"` field and run by the generic `_apply_wiring_entry` executor (bound
at import by `_bind_wiring_patches`). The remaining **22** are *outliers* that keep a hand-written
`@register_patch apply_patch_*` function because they carry real apply-time logic (kernel installs,
class rebinds, bundled preallocs, hardcoded skips). This deleted ~85 near-identical ~36-line
functions, taking `apply_all.py` from ~4,800 to ~1,900 lines. Boot order is no longer implicit
source-line order — it's the explicit `_APPLY_ORDER` list (both paths read their `_apply_order`
from it). Tests assert the **seam** (`PATCH_REGISTRY["<ID>"]["wiring"]` + a bound `apply_callable`),
not a scaffolding function name.

### Patch IDs and lifecycle

- **Three ID schemes**: `PR<prnum>` for a patch backed by an upstream vLLM PR (e.g. `PR40898`;
  same-PR collisions get a lowercase suffix, `PR40738`/`PR40738b`); `PN<NN>` for Genesis-original
  "new series" patches with no upstream PR; `P<NN>` for legacy pre-dispatcher patches. The `PR####`
  rename happened 2026-06 — older docs/commits still reference the pre-rename `PN21`/`P62` ids.
- **Env flags carry the id + an intent suffix**: `GENESIS_ENABLE_<ID>_<DESCRIPTION>`, e.g.
  `GENESIS_ENABLE_PR40898_DFLASH_SWA` (uppercase id even for sub-patches:
  `GENESIS_ENABLE_PR40738B_TRITON_KERNEL`). The handful of PR-backed patches that never had a
  descriptive name stay bare (`GENESIS_ENABLE_PR40385`). Pre-rename `GENESIS_ENABLE_*` names survive
  in an `env_flag_aliases` list — `should_apply` still honors them with a one-time deprecation
  warning — but the bare `GENESIS_ENABLE_PR#####` form is **not** kept as an alias (the 2026-06
  "restore intent to the suffix" pass put the description back onto the canonical and dropped it).
- For the ~22 outlier functions the apply function name encodes the id (`_APPLY_PATCH_ID_RE`):
  `apply_patch_pr40898_*` → `PR40898`, `apply_patch_N21_*` → `PN21`, `apply_patch_67_*` → `P67`
  (`register_patch(..., patch_id=...)` can override). Metadata-driven patches carry no function —
  their id is the registry key and their implementation is the `wiring` field's module stem.
- Patches default **OFF** (`default_on: False`), opt-in via their env flag. Global opt-out:
  `GENESIS_DISABLE=1`.
- A patch self-retires when `upstream_drift_markers` / `upstream_compat` detect the fix landed
  upstream. **Curation policy:** patches whose upstream PR is *closed-unmerged* are removed
  outright (13 pruned in 2026-06: a batch of 12 + `PR39598`); ones whose PR has *merged* are
  retirement candidates.

## Common commands

```bash
# Run the full test suite. ALWAYS use `python3 -m pytest`, never bare `pytest` —
# the vllm/_genesis namespace package needs repo-root on sys.path at launch time
# (bare pytest fails to resolve `vllm._genesis` on macOS; works by luck on Linux).
python3 -m pytest vllm/_genesis/tests/ -q

# A single test file / single test
python3 -m pytest vllm/_genesis/tests/test_pn59_streaming_gdn.py -v
python3 -m pytest vllm/_genesis/tests/test_dispatcher_validator.py::test_<name> -v

# Skip GPU-only tests (markers: gpu, integration, slow — see pytest.ini)
python3 -m pytest vllm/_genesis/tests/ -m 'not gpu'

# The three CI gates beyond pytest (all exit 1 on failure):
python3 -m vllm._genesis.compat.lifecycle_audit_cli --quiet   # patch lifecycle states
python3 -m vllm._genesis.compat.schema_validator              # PATCH_REGISTRY shape
python3 -m vllm._genesis.compat.cli self-test --quiet         # structural sanity

# Diagnostics (no vLLM boot required)
python3 -m vllm._genesis.dispatcher           # per-patch apply/skip decision matrix
python3 -m vllm._genesis.patches.apply_all    # dry orchestrator run + boot summary
python3 -m vllm._genesis.compat.cli doctor    # full hw+sw+model+patch diagnostic
python3 -m vllm._genesis.compat.cli explain P67   # one patch in detail
```

The `genesis <subcommand>` shorthand (installed by `install.sh`) is a thin wrapper over
`python3 -m vllm._genesis.compat.cli <subcommand>`. **torch is a runtime-only dependency** (as of
2026-06): the core package imports and the full suite runs with **no torch installed** — kernel /
numeric tests `pytest.importorskip("torch")` and skip cleanly. `pytest` is the only test dep. (One
pre-existing failure, `test_default_dir_under_user_home`, appears in sandboxes with a read-only
`~/.cache` — unrelated to torch.)

## Conventions when changing code

- **МЫ ЧИНИМ, НЕ ЛОМАЕМ ("we fix, we don't break").** `apply()` and `apply_patch_*` must never
  raise — wrap in `try/except` and return `("failed", str(e))` / a `failed` `PatchResult`. A bad
  patch must never crash engine boot.
- **Anchors are VERBATIM upstream source.** Copy-paste the exact lines (including their
  indentation/whitespace) into a `TextPatch.anchor`. Never reformat or normalize while patching.
- **Lazy `vllm.*` imports inside functions**, never at module top level — a different vLLM pin may
  have renamed the module, and top-level imports would break boot / test collection.
- **torch is runtime-only** — never `import torch` at module top in package code; guard it
  (`try: import torch\nexcept ModuleNotFoundError: torch = None`) or import lazily inside functions,
  so the package imports without torch. torch does **not** affect text-patch application (proven by
  patches applying against a torch-less vLLM tree) — it's only needed by the `kernels/*` compute
  modules at inference time. The 4 pure-compute kernels still needing torch to import are the known
  exception.
- **Markers include the version**: `Genesis <ID> v7.NN_descriptive_name` (`<ID>` = the patch id,
  e.g. `PR40898` / `PN59`).
- **Detection logic goes in `guards.py`** — don't re-implement vendor/SM/arch checks in a patch.
- Conventional commits with a patch- or subsystem-scoped scope, e.g. `fix(PN59): ...`,
  `feat(patch): P88 ...`, `perf(kernel): ...`, `test(P68): ...`, `feat(model_configs): ...`,
  `release(v7.72.5): ...`. One logical patch per commit. Allowed types observed in history:
  `feat`, `fix`, `docs`, `perf`, `test`, `chore`, `refactor`, `ci`, `release`, `revert`, `security`.
- **`git commit -am` skips newly-added files** (test files have been lost this way before — see
  commit `41b13a6`). Stage new files explicitly with `git add` before committing.
- Commit subjects close issues in **community reproducer repos**, not this repo's tracker — e.g.
  `closes club-3090#22`, `noonghunna/club-3090#57`. Keep that cross-repo `<repo>#<n>` form.
- **"Cliff N"** is project jargon for a known perf/correctness wall (e.g. Cliff 1 = FFN cache,
  Cliff 2b = multi-turn long-context OOM). Catalogued in `docs/CLIFFS.md`; reference by number.
- Large fixes land incrementally as "**Level 1/2/3**" (or "Phase 1/2") sub-commits under one patch
  — partial skeletons are committed and noted as such rather than held back.

## Repository workflow (from git history)

- **Branching:** feature/fix work lands on the **`dev`** branch; a release then merges `dev` →
  `main` with a `release(vX.Y.Z): merge dev into main — <summary>` commit. `main` is the released,
  tagged line — base new work on `dev` unless a change is itself a release/hotfix to `main`.
- **Versioning:** `v7.NN[.x]` (currently `v7.72.5`); frequent point releases. Stable tags may
  carry a date suffix (`v7.51-stable-2026-04-27`). Two changelogs are maintained: `CHANGELOG.md`
  (public, per-release) and `vllm/_genesis/CHANGELOG.md` (engineering, per-commit/per-A·B).
- Sole maintainer history (Sander / Александр Барзов); Dependabot handles GitHub Actions bumps.

## Adding a patch (the core dev loop)

1. Create `vllm/_genesis/wiring/<category>/patch_<id>_<name>.py` with an `apply() -> (status, reason)`
   using `TextPatcher`/`TextPatch`. Use `pr<prnum>` in the filename if the patch backports an
   upstream PR, else `N<NN>`. Categories: `spec_decode`, `structured_output`, `kv_cache`,
   `kernels`, `compile_safety`, `perf_hotfix`, `hybrid`, `middleware`, `loader`, `memory`,
   `legacy` (default to `perf_hotfix` if unsure).
2. Add the metadata entry to `dispatcher.py::PATCH_REGISTRY` (`default_on: False`; set `upstream_pr`
   + `env_flag = GENESIS_ENABLE_<ID>_<DESCRIPTION>`). **For the common case — a pure text-patch/rebind dispatch —
   also set `wiring: "patch_<id>_<name>"` (the module stem) and you're done: the generic
   `_apply_wiring_entry` executor runs it, no apply function needed.** This single entry is the
   source of truth.
3. Add the patch id to `_APPLY_ORDER` in `apply_all.py` at the position it should boot (omit it and
   it runs last). That list is the explicit boot order.
4. **Only if the patch needs real apply-time logic** beyond dispatch (kernel install, conditional
   rebind, multi-module orchestration): skip the `wiring` field and instead write a
   `@register_patch("<display>") apply_patch_<id>_*` *outlier* function in `apply_all.py` — it
   auto-attaches its callable onto the dispatcher entry.
5. Add `vllm/_genesis/tests/test_<id>_<name>.py` (TDD: write it first). Minimum coverage: anchor
   exists in the pinned vLLM source, replacement is well-formed, marker present, `apply()` is
   idempotent, and the seam is wired (`PATCH_REGISTRY["<ID>"]["wiring"]` + a bound `apply_callable`).
   If it exercises torch, `pytest.importorskip("torch")` at the top.
6. Run the suite + the three CI gates above.

Full contributor guide with PR template: `docs/CONTRIBUTING.md`. Engineering README for the
package internals: `vllm/_genesis/README.md`. Patch catalog: `docs/PATCHES.md`.

## vLLM pin

Patches text-edit specific upstream files at known anchors, so they are tightly coupled to the
pinned vLLM version (currently `0.20.2rc1.dev9+g01d4d1ad3`). If the pin drifts, anchors stop
matching and patches `SKIPPED (anchor not found)` — a boot summary full of those skips means the
pin moved, not that patches are broken. See `docs/COMPATIBILITY.md` and `tools/check_upstream_drift.py`.
