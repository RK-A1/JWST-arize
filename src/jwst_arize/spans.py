"""
Rebuilding an agent turn from its spans.

Production scoring has no return value to grade — only a trace. This module
turns a Phoenix span dataframe back into the same `{answer, trajectory}` shape
the agent returns in-process, so the scorers in evaluators.py can grade
production traffic and offline experiments with one implementation.

Shared by `scripts/score_traffic.py` and `scripts/curate_failures.py`, which is
the point: if reconstruction drifted between the two, the dataset curated from
failures would not contain the failures that were measured.
"""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

import pandas as pd

ROOT_SPAN_NAME = "agent-turn"


def photo_ids(value: Any) -> list[str]:
    """Pull photo_ids out of a tool span's output payload, whatever its nesting."""
    found: list[str] = []

    def walk(node: Any) -> None:
        if isinstance(node, list):
            for item in node:
                walk(item)
        elif isinstance(node, dict):
            for key, child in node.items():
                if key == "photo_id" and isinstance(child, str):
                    found.append(child)
                else:
                    walk(child)
        elif isinstance(node, str):
            try:
                parsed = json.loads(node)
            except (ValueError, TypeError):
                return
            walk(parsed)

    walk(value)
    return list(dict.fromkeys(found))


def final_answer(output_value: Any) -> str:
    """The last assistant message in the root span's output."""
    try:
        payload = json.loads(output_value) if isinstance(output_value, str) else output_value
        messages = payload.get("messages", [])
    except (ValueError, TypeError, AttributeError):
        return ""
    for msg in reversed(messages):
        data = msg.get("data", {}) if isinstance(msg, dict) else {}
        if msg.get("type") == "ai" and data.get("content"):
            content = data["content"]
            if isinstance(content, list):
                return " ".join(b.get("text", "") for b in content if isinstance(b, dict))
            return str(content)
    return ""


def reconstruct(df: pd.DataFrame) -> list[dict[str, Any]]:
    """One record per trace: case_id, answer, trajectory, and the root span id.

    The case_id is read from span metadata rather than the root span, because
    LangGraph overwrites metadata on the span it owns. Any span in the trace
    carries it, so the join is per-trace rather than per-span.
    """
    by_trace: dict[str, list[Any]] = defaultdict(list)
    for _, row in df.iterrows():
        by_trace[row["context.trace_id"]].append(row)

    records: list[dict[str, Any]] = []
    for trace_id, spans in by_trace.items():
        root = next((s for s in spans if s["name"] == ROOT_SPAN_NAME), None)
        if root is None:
            continue

        case_id = None
        for s in spans:
            md = s.get("attributes.metadata")
            if isinstance(md, dict) and md.get("case_id"):
                case_id = md["case_id"]
                break
        if case_id is None:
            continue

        records.append(
            {
                "trace_id": trace_id,
                "span_id": root["context.span_id"],
                "case_id": case_id,
                "answer": final_answer(root.get("attributes.output.value")),
                "trajectory": [
                    {"name": s["name"], "returned_photo_ids": photo_ids(s.get("attributes.output.value"))}
                    for s in spans
                    if s.get("span_kind") == "TOOL"
                ],
            }
        )
    return records
