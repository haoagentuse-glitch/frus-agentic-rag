"""Streamlit trace UI.

The point of the UI is not chat: it is showing what the agent decided. Route,
subqueries, tool calls, which evidence the grader accepted, whether a
correction fired, and the citation gate's verdict are all on screen, because a
system whose selling point is bounded decisions has to make them inspectable.
"""

from __future__ import annotations

import os

import httpx
import streamlit as st

API = os.getenv("FRUS_API_URL", "http://localhost:8000")

st.set_page_config(page_title="FRUS Agentic RAG", page_icon="📜", layout="wide")

SYSTEM_HELP = {
    "B0-2step": "Fixed 2-step baseline: retrieve once, generate, validate.",
    "B0": "Graph wiring of the baseline: no planner, no grader.",
    "B1": "B0 + planning and decomposition.",
    "B2": "B1 + evidence grading, rewrite and abstain.",
    "B3": "B2 + citation gate and repair. Full system.",
}


@st.cache_data(ttl=30)
def health() -> dict:
    try:
        return httpx.get(f"{API}/health", timeout=10).json()
    except Exception as exc:
        return {"status": "unreachable", "error": str(exc)}


with st.sidebar:
    st.title("📜 FRUS Agentic RAG")
    h = health()
    if h.get("status") == "ok":
        st.success("API ready")
    else:
        st.error(f"API {h.get('status')}")
    idx = h.get("index", {})
    if "rows" in idx:
        st.metric("Indexed chunks", f"{idx['rows']:,}")
        st.metric("With dense vectors", f"{idx.get('embedded', 0):,}")
        if idx.get("embedded", 0) == 0:
            st.caption("Dense index still building — retrieval is BM25-only for now.")
    system = st.selectbox("System", list(SYSTEM_HELP), index=4)
    st.caption(SYSTEM_HELP[system])
    language = st.radio("Answer language", ["zh-TW", "en"], horizontal=True)
    st.divider()
    st.caption(
        "Answers cite FRUS documents or abstain. No web search, no answering from model memory."
    )

st.header("Ask the FRUS corpus")

examples = [
    "尼克森政府在 1969 到 1972 年間對中國政策的立場如何演變？",
    "What did the United States tell Britain about recognition of the Confederacy in 1862?",
    "Is the FRUS volume covering the 2003 Iraq war published yet?",
]
cols = st.columns(len(examples))
for i, ex in enumerate(examples):
    if cols[i].button(ex[:40] + "…", key=f"ex{i}", use_container_width=True):
        st.session_state["question"] = ex

question = st.text_area("Question", key="question", height=90)

if st.button("Ask", type="primary") and question.strip():
    with st.spinner(f"Running {system}…"):
        try:
            r = httpx.post(
                f"{API}/ask",
                json={"question": question, "language": language, "system": system},
                timeout=300,
            )
            r.raise_for_status()
            data = r.json()
        except Exception as exc:
            st.error(f"Request failed: {exc}")
            st.stop()

    outcome = data["outcome"]
    if outcome == "abstain":
        st.warning("Abstained — the corpus does not support a grounded answer.")
        if data.get("abstain_reason"):
            st.caption(f"Reason: {data['abstain_reason']}")
        st.info(data.get("answer_text", ""))
    else:
        st.markdown(data["answer_text"])

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Outcome", outcome)
    m2.metric("LLM calls", data["llm_calls"])
    m3.metric("Retrievals", data["retrieval_calls"])
    m4.metric("Latency", f"{data['latency_s']:.1f}s")

    if data.get("citations"):
        st.subheader("Citations")
        for c in data["citations"]:
            st.markdown(f"- {c}")

    if data.get("limitations"):
        st.caption(f"Limitations: {data['limitations']}")

    st.subheader("Graph path")
    trace = data.get("trace", [])
    st.code(" → ".join(t["node"] for t in trace) or "(no trace)", language="text")

    for t in trace:
        detail = t.get("detail", {})
        label = f"{t['node']} · {t.get('duration_s', 0):.2f}s"
        if t.get("error"):
            label += " · ⚠"
        with st.expander(label):
            if t["node"] == "plan_query":
                st.write(
                    f"**Route:** `{detail.get('route')}` "
                    f"(rule-first would say `{detail.get('rule_route')}`)"
                )
                for q in detail.get("subqueries", []):
                    st.write(f"- {q}")
            elif t["node"] == "dispatch_retrieval":
                st.write(
                    f"**Tool:** `{detail.get('tool')}` · hops {detail.get('hops')} "
                    f"· {detail.get('hits', 0)} hits"
                )
            elif t["node"] == "grade_evidence":
                st.write(
                    f"**Verdict:** `{detail.get('overall')}` · {detail.get('accepted', 0)} accepted"
                )
                if detail.get("hallucinated_ids"):
                    st.error(f"Grader invented ids: {detail['hallucinated_ids']}")
                for hop in detail.get("hops", []):
                    st.write(f"- `{hop.get('verdict')}` {hop.get('corrective_query') or ''}")
            elif t["node"] == "rewrite_missing":
                st.write("**Correction fired.** Rewritten hops:")
                for q in detail.get("rewritten", []):
                    st.write(f"- {q}")
            elif t["node"] == "validate_citations":
                if detail.get("errors"):
                    st.error("\n".join(f"- {e}" for e in detail["errors"]))
                else:
                    st.success(f"{detail.get('cited_documents', 0)} documents validated")
            else:
                st.json(detail)
            if t.get("error"):
                st.warning(t["error"])

    if data.get("evidence"):
        st.subheader(f"Accepted evidence ({len(data['evidence'])})")
        for e in data["evidence"]:
            with st.expander(f"{e['evidence_id']} · {e['date_from']} · {e['head'][:70]}"):
                st.caption(
                    f"{e['volume_id']} Doc. {e['doc_number']} · hop `{e.get('hop', '')}` · "
                    f"bm25 {e.get('rank_bm25')} · dense {e.get('rank_dense')}"
                )
                st.write(e["text"][:2500])
                st.markdown(
                    f"[history.state.gov]"
                    f"(https://history.state.gov/historicaldocuments/"
                    f"{e['volume_id']}/{e['document_id']})"
                )
