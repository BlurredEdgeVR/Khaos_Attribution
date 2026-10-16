"""The segment-id rule: ``<track_id>_seg_<index>`` with any crop or stem suffix after it. Written once; the rooms
re-export it so a pair file's name always says which track it came from."""

from __future__ import annotations


def make_segment_id(track_id: str, index: int) -> str:
    return f"{track_id}_seg_{index:03d}"


def track_of_segment(segment_id: str) -> str:
    """The track a segment id belongs to: everything before the last ``_seg_``; a string with none is itself."""
    head, sep, _ = str(segment_id).rpartition("_seg_")
    return head if sep else str(segment_id)
