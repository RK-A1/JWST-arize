"""
The JWST research agent — a LangGraph ReAct loop over three corpus tools.

One call to `run_agent` produces one OpenInference trace:

    AGENT   agent-turn
    ├── LLM   ChatOpenAI            (auto-captured: prompt, tokens, cost)
    ├── TOOL  search_photos
    ├── TOOL  get_photo
    ├── LLM   ChatOpenAI
    └── ...

Nothing in this file mentions Phoenix or Arize. The instrumentation is the
LangChainInstrumentor registered in tracing.py, so the agent is written the way
the team would have written it anyway and the observability is bolted on
underneath. That is the honest version of "no code changes to instrument" —
worth being able to show, because prospects ask.

`run_agent` also returns a structured `trajectory`. Evaluators can read it
directly (fast, deterministic, offline) or read the same information back out of
the span tree (which is what online scoring against production has to do). Both
paths are exercised in this repo, deliberately.
"""

from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from .corpus import count_by_label, get_photo, search_photos
from .prompts import system_prompt

DEFAULT_MODEL = os.getenv("AGENT_MODEL", "claude-haiku-4-5")
DEFAULT_MAX_STEPS = 8

TOOL_NAMES = ("search_photos", "get_photo", "count_by_label")


# ── Tools ───────────────────────────────────────────────────────────────────
# Docstrings are the tool descriptions the model sees, so they are written for
# the model rather than for a reader of this file.


@tool
def search_photos_tool(query: str, limit: int = 8) -> dict[str, Any]:
    """Search photo titles, descriptions, and tags. Returns ranked matches with
    their IDs and labels — summaries only. Call get_photo for the full record,
    including the capture date."""
    return {"hits": search_photos(query, min(limit, 25))}


@tool
def get_photo_tool(photo_id: str) -> dict[str, Any]:
    """Retrieve the complete record for one photo by ID: title, description,
    tags, canonical label, capture date, and image URL. This is the only way to
    confirm that a photo ID actually exists."""
    photo = get_photo(photo_id)
    # A miss is a signal, not an exception: it means the agent asked for an ID
    # that does not exist, which is exactly what the grounding evaluator is for.
    if photo is None:
        return {"found": False, "photo_id": photo_id, "error": f"No photo with ID {photo_id}."}
    return {"found": True, **photo}


@tool
def count_by_label_tool() -> dict[str, Any]:
    """Exact photo count for every canonical label in the corpus. Use this for
    any 'how many' question — do not estimate from search results."""
    counts = count_by_label()
    return {"total": sum(counts.values()), "counts": counts}


# LangChain derives the tool name from the function name, so rename to the
# stable names the prompts and evaluators refer to.
search_photos_tool.name = "search_photos"
get_photo_tool.name = "get_photo"
count_by_label_tool.name = "count_by_label"

TOOLS = [search_photos_tool, get_photo_tool, count_by_label_tool]


# ── Result shape ────────────────────────────────────────────────────────────


@dataclass
class ToolCallRecord:
    name: str
    args: dict[str, Any]
    returned_photo_ids: list[str]
    duration_ms: int = 0


@dataclass
class AgentResult:
    answer: str
    trajectory: list[ToolCallRecord] = field(default_factory=list)
    steps: int = 0
    truncated: bool = False
    model: str = DEFAULT_MODEL
    prompt_version: str = "v1-grounded"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _extract_photo_ids(value: Any) -> list[str]:
    """Pull every photo_id a tool result mentions, at any nesting depth."""
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

    walk(value)
    return list(dict.fromkeys(found))


def _model(model: str) -> ChatOpenAI:
    """An OpenAI-compatible client.

    Points at whatever OPENAI_BASE_URL is set to, so the same agent runs against
    OpenAI directly, against an Anthropic model through a gateway, or against a
    self-hosted endpoint. The provider is not the interesting variable here.
    """
    kwargs: dict[str, Any] = {"model": model, "temperature": 0}
    base_url = os.getenv("OPENAI_BASE_URL")
    if base_url:
        kwargs["base_url"] = base_url
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        kwargs["api_key"] = api_key
    return ChatOpenAI(**kwargs)


_GRAPH_CACHE: dict[tuple[str, str], Any] = {}


def _graph(prompt_version: str, model: str) -> Any:
    key = (prompt_version, model)
    if key not in _GRAPH_CACHE:
        _GRAPH_CACHE[key] = create_react_agent(
            _model(model),
            TOOLS,
            prompt=SystemMessage(content=system_prompt(prompt_version)),
            name="agent-turn",
        )
    return _GRAPH_CACHE[key]


def run_agent(
    question: str,
    prompt_version: str = "v1-grounded",
    model: str = DEFAULT_MODEL,
    max_steps: int = DEFAULT_MAX_STEPS,
) -> AgentResult:
    """Run one question through the agent and return its answer and trajectory."""
    graph = _graph(prompt_version, model)
    started = time.perf_counter()

    try:
        state = graph.invoke(
            {"messages": [("user", question)]},
            config={"recursion_limit": max_steps * 2},
        )
    except Exception:  # noqa: BLE001
        # A crash mid-run is a real production failure mode. Record it as an
        # empty answer rather than losing the row — an eval suite that silently
        # drops its hardest cases reports a flattering number. The turn is
        # marked truncated, which scores badly, which is the correct outcome.
        return AgentResult(
            answer="",
            trajectory=[],
            steps=0,
            truncated=True,
            model=model,
            prompt_version=prompt_version,
            )

    messages = state["messages"]
    trajectory: list[ToolCallRecord] = []
    pending: dict[str, dict[str, Any]] = {}

    for msg in messages:
        if isinstance(msg, AIMessage):
            for call in msg.tool_calls or []:
                pending[call["id"]] = {"name": call["name"], "args": call.get("args", {})}
        elif isinstance(msg, ToolMessage):
            meta = pending.pop(msg.tool_call_id, {"name": msg.name, "args": {}})
            trajectory.append(
                ToolCallRecord(
                    name=meta["name"],
                    args=meta["args"],
                    returned_photo_ids=_extract_photo_ids(_maybe_json(msg.content)),
                )
            )

    final = messages[-1]
    answer = final.content if isinstance(final, AIMessage) else ""
    if isinstance(answer, list):  # content blocks
        answer = " ".join(b.get("text", "") for b in answer if isinstance(b, dict))

    elapsed = int((time.perf_counter() - started) * 1000)
    if trajectory:
        trajectory[-1].duration_ms = elapsed

    return AgentResult(
        answer=answer or "",
        trajectory=trajectory,
        steps=sum(1 for m in messages if isinstance(m, AIMessage)),
        truncated=not answer,
        model=model,
        prompt_version=prompt_version,
    )


def _maybe_json(content: Any) -> Any:
    import json

    if isinstance(content, str):
        try:
            return json.loads(content)
        except (ValueError, TypeError):
            return content
    return content
