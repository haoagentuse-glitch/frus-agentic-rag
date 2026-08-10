"""Local Gradio chat over the FRUS corpus.

The Streamlit app in `ui.py` is a trace inspector; this one is the plain
question-and-answer surface. It calls the graph in-process instead of going
through the FastAPI service, so showing the system to someone takes one
command and no separately running server.
"""

from __future__ import annotations

import gradio as gr

from frus_agentic_rag.models import Answer

# B3 first because it is the full system and the sensible default; the rest are
# the ablation arms, kept selectable so the difference can be shown live.
SYSTEMS = ["B3", "B0-2step", "B0", "B1", "B2"]
LANGUAGES = ["auto", "zh-TW", "en"]

DESCRIPTION = (
    "Answers cite FRUS documents or abstain. No web search, no answering from model memory."
)


def _route(trace: list[dict]) -> str:
    """The route lives in the planner's trace event, and B0-2step never plans."""
    for event in trace:
        if event.get("node") == "plan_query":
            return str(event.get("detail", {}).get("route") or "-")
    return "-"


def _summary(result: Answer) -> str:
    """The bounded-decision claim is only credible if the counters are on screen."""
    return (
        f"`outcome={result.outcome}` · `system={result.system}` · `route={_route(result.trace)}` "
        f"· `llm_calls={result.llm_calls}` · `retrieval_calls={result.retrieval_calls}` "
        f"· `{result.latency_s:.1f}s`"
    )


def _render(result: Answer) -> str:
    """Abstention is a correct outcome for this system, so it renders as a result."""
    if result.outcome == "abstain":
        parts = [
            "**Abstained — no answer given.** "
            "The retrieved corpus did not support a grounded answer.",
            f"Reason: {result.abstain_reason or 'no grounded evidence'}",
        ]
    else:
        parts = [result.answer_text or "_(empty answer)_"]
        if result.limitations:
            parts.append(f"_Limitations: {result.limitations}_")
    if result.citations:
        # The citation lines already carry the canonical history.state.gov URL,
        # which only code is allowed to build — never reformat them here.
        parts.append("**Citations**\n" + "\n".join(f"- {line}" for line in result.citations))
    parts.append(_summary(result))
    return "\n\n".join(parts)


async def respond(message: str, history: list[dict], system: str, language: str) -> str:
    """Async so Gradio awaits the graph on its own loop; asyncio.run would fail here."""
    from frus_agentic_rag.agent.run import answer, answer_2step

    question = message.strip()
    if not question:
        return "Ask a question about the FRUS corpus."
    # "auto" means no override: detect_language reads the question instead.
    lang = None if language == "auto" else language
    try:
        if system == "B0-2step":
            # The fixed baseline is not a graph wiring, so it has its own entry point.
            result = await answer_2step(question, language=lang)
        else:
            result = await answer(question, language=lang, system=system)
    except Exception as exc:
        # Kept visually distinct from an abstention: one is a bug, the other is
        # the system working as designed.
        return (
            "**Run failed** — this is an error, not an abstention.\n\n"
            f"`{type(exc).__name__}: {exc}`"
        )
    return _render(result)


def build() -> gr.Blocks:
    """Returns the app unlaunched, so a smoke check can construct it without serving."""
    system = gr.Dropdown(SYSTEMS, value="B3", label="System")
    language = gr.Dropdown(LANGUAGES, value="auto", label="Answer language")
    return gr.ChatInterface(
        respond,
        title="📜 FRUS Agentic RAG",
        description=DESCRIPTION,
        additional_inputs=[system, language],
        # This app is local-only; the constructor flag also suppresses gradio's
        # version check on launch.
        analytics_enabled=False,
    )
