# SPDX-License-Identifier: Apache-2.0
"""Wiring for PR44283 — vllm#44283 backport: Anthropic system-role messages.

Backport of upstream vllm-project/vllm#44283 (chaunceyjiang, MERGED
2026-06-02). Fixes vllm#44000.

## What it does

vLLM's Anthropic-compatible endpoint (`/v1/messages`) only accepted a
top-level `system` field. A `{"role": "system", ...}` entry *inside* the
`messages` array — which several Anthropic SDK clients emit — was rejected
by the `AnthropicMessage.role` `Literal["user", "assistant"]` validator,
or (where validation was looser) silently passed through `_convert_messages`
as an `openai_msg` with `role="system"` mid-conversation.

The backport makes two coordinated edits:

1. `entrypoints/anthropic/protocol.py` — widen `AnthropicMessage.role` to
   also allow `"system"`.
2. `entrypoints/anthropic/serving.py`:
   - `_convert_system_message` — collect system text from BOTH the
     top-level `system` field AND any `role == "system"` entries in the
     messages array, concatenating into a single OpenAI system message.
   - `_convert_messages` — skip `role == "system"` entries (already folded
     into the system message above) so they are not emitted twice.

## Why this is a Genesis patch

Endpoint correctness fix. The Genesis pin (`g01d4d1ad3`) predates the
2026-06-02 merge, so the anchors below are present verbatim. Once the pin
advances past the merge the post-fix source already contains the new code;
the per-file `upstream_drift_markers` detect that and the patch SKIPS
(auto-retires) — it never double-applies.

Multi-file (protocol.py + serving.py = 2 files, 3 sub-patches), applied
atomically via `MultiFilePatchTransaction` (validate-all-then-write-all,
true rollback on a commit-phase race). Default OFF; opt-in via
`GENESIS_ENABLE_PR44283_ANTHROPIC_SYSTEM_ROLE`. Model-agnostic (endpoint
layer), so no `applies_to` restriction.

Author: chaunceyjiang (vllm#44283); Genesis backport ToastyTheBot.
"""
from __future__ import annotations

import logging

from vllm._genesis.guards import resolve_vllm_file, vllm_install_root
from vllm._genesis.wiring.text_patch import (
    TextPatch,
    TextPatcher,
)

log = logging.getLogger("genesis.wiring.pr44283_anthropic_system_role")

GENESIS_PR44283_MARKER = (
    "Genesis PR44283 Anthropic system-role in messages array (vllm#44283)"
)


# ─── Sub-A: protocol.py — widen AnthropicMessage.role to allow "system" ─────
PROTOCOL_OLD = (
    "    role: Literal[\"user\", \"assistant\"]\n"
    "    content: str | list[AnthropicContentBlock]"
)
PROTOCOL_NEW = (
    "    role: Literal[\"user\", \"assistant\", \"system\"]\n"
    "    content: str | list[AnthropicContentBlock]"
)


# ─── Sub-B: serving.py — _convert_system_message collects from both sources ─
SERVING_SYS_OLD = (
    "        if not anthropic_request.system:\n"
    "            return\n"
    "\n"
    "        if isinstance(anthropic_request.system, str):\n"
    "            openai_messages.append(\n"
    "                {\"role\": \"system\", \"content\": anthropic_request.system}\n"
    "            )\n"
    "        else:\n"
    "            system_prompt = \"\"\n"
    "            for block in anthropic_request.system:\n"
    "                if block.type == \"text\" and block.text:\n"
    "                    # Strip Claude Code's attribution header which contains\n"
    "                    # a per-request hash that defeats prefix caching.\n"
    "                    if block.text.startswith(\"x-anthropic-billing-header\"):\n"
    "                        continue\n"
    "                    system_prompt += block.text\n"
    "            openai_messages.append({\"role\": \"system\", \"content\": system_prompt})"
)
SERVING_SYS_NEW = (
    "        # [Genesis PR44283 vllm#44283] collect system text from the top-level\n"
    "        # `system` field AND any role==\"system\" entries in the messages array.\n"
    "        system_parts: list[str] = []\n"
    "\n"
    "        # Top-level system field\n"
    "        if anthropic_request.system:\n"
    "            if isinstance(anthropic_request.system, str):\n"
    "                system_parts.append(anthropic_request.system)\n"
    "            else:\n"
    "                for block in anthropic_request.system:\n"
    "                    if block.type == \"text\" and block.text:\n"
    "                        # Strip Claude Code's attribution header which contains\n"
    "                        # a per-request hash that defeats prefix caching.\n"
    "                        if block.text.startswith(\"x-anthropic-billing-header\"):\n"
    "                            continue\n"
    "                        system_parts.append(block.text)\n"
    "\n"
    "        # System messages embedded inside the messages array\n"
    "        for msg in anthropic_request.messages:\n"
    "            if msg.role != \"system\":\n"
    "                continue\n"
    "            if isinstance(msg.content, str):\n"
    "                system_parts.append(msg.content)\n"
    "            else:\n"
    "                for block in msg.content:\n"
    "                    if block.type == \"text\" and block.text:\n"
    "                        if block.text.startswith(\"x-anthropic-billing-header\"):\n"
    "                            continue\n"
    "                        system_parts.append(block.text)\n"
    "\n"
    "        if system_parts:\n"
    "            openai_messages.append({\"role\": \"system\", \"content\": \"\".join(system_parts)})"
)


# ─── Sub-C: serving.py — _convert_messages skips system-role entries ────────
SERVING_MSG_OLD = (
    "        \"\"\"Convert Anthropic messages to OpenAI format\"\"\"\n"
    "        for msg in messages:\n"
    "            openai_msg: dict[str, Any] = {\"role\": msg.role}  # type: ignore"
)
SERVING_MSG_NEW = (
    "        \"\"\"Convert Anthropic messages to OpenAI format\"\"\"\n"
    "        for msg in messages:\n"
    "            # [Genesis PR44283 vllm#44283] system-role folded into system message\n"
    "            if msg.role == \"system\":\n"
    "                continue\n"
    "\n"
    "            openai_msg: dict[str, Any] = {\"role\": msg.role}  # type: ignore"
)


def _make_protocol_patcher() -> TextPatcher | None:
    target = resolve_vllm_file("entrypoints/anthropic/protocol.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name="PR44283 protocol.py (AnthropicMessage.role)",
        target_file=str(target),
        marker=GENESIS_PR44283_MARKER + " (protocol)",
        sub_patches=[
            TextPatch(
                name="pr44283_role_literal",
                anchor=PROTOCOL_OLD,
                replacement=PROTOCOL_NEW,
                required=True,
            ),
        ],
        upstream_drift_markers=[
            "role: Literal[\"user\", \"assistant\", \"system\"]",
        ],
    )


def _make_serving_patcher() -> TextPatcher | None:
    target = resolve_vllm_file("entrypoints/anthropic/serving.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name="PR44283 serving.py (system-message conversion)",
        target_file=str(target),
        marker=GENESIS_PR44283_MARKER + " (serving)",
        sub_patches=[
            TextPatch(
                name="pr44283_convert_system_message",
                anchor=SERVING_SYS_OLD,
                replacement=SERVING_SYS_NEW,
                required=True,
            ),
            TextPatch(
                name="pr44283_convert_messages_skip_system",
                anchor=SERVING_MSG_OLD,
                replacement=SERVING_MSG_NEW,
                required=True,
            ),
        ],
        upstream_drift_markers=[
            "# System messages embedded inside the messages array",
        ],
    )


def apply() -> tuple[str, str]:
    """Apply PR44283 wiring (2 files, 3 sub-patches). Never raises.

    Atomic via MultiFilePatchTransaction: either protocol.py + serving.py
    both commit, or neither does. Idempotent + auto-no-op once #44283 is in
    the pinned vLLM source.
    """
    from vllm._genesis.dispatcher import log_decision, should_apply

    decision, reason = should_apply("PR44283")
    log_decision("PR44283", decision, reason)
    if not decision:
        return "skipped", reason

    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    from vllm._genesis.wiring.text_patch import MultiFilePatchTransaction

    raw = [_make_protocol_patcher(), _make_serving_patcher()]
    if any(p is None for p in raw):
        return "skipped", "Anthropic endpoint (entrypoints/anthropic/*) not found"
    patchers = [p for p in raw if p is not None]

    txn = MultiFilePatchTransaction(patchers, name="PR44283")
    return txn.apply_or_skip()
