"""Haan processor -- subclasses the vendored Moshi processor.

Inherits MoshiProcessor (Mimi codec + text tokenizer) from
`project_amnesty.models.moshi`. TODO(ARCH 7): use the Qwen3 tokenizer, and add the
Zone A/B/C instruction-template layout + PAD/EPAD reserved slots (7.6).
"""

from __future__ import annotations

from project_amnesty.models.moshi import MoshiProcessor

__all__ = ["HaanProcessor"]


class HaanProcessor(MoshiProcessor):
    """Mimi codec + text tokenizer + Zone/PAD layout (ARCHITECTURE 7). Inherits MoshiProcessor."""
