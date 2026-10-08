"""Provable, traceable, verifiable legal-assistant interface."""
from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from ingestion.chunker import chunk_pages
from ingestion.pdf_loader import load_documents
from retrieval.hybrid_search import HybridRetriever, configure

from app.backend_adapter import (
    answer_with_grounding,
    normalize_answer,
    retrieve,
    review_case,
)

st.set_page_config(
    page_title="Agentic Legal Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.block-container {padding-top: 2rem; padding-bottom: 3rem;}
.hero {padding: 1.2rem 1.4rem; border: 1px solid rgba(128,128,128,.25);
       border-radius: 14px; margin-bottom: 1rem;}
.badge {display:inline-block; padding:.2rem .55rem; border-radius:999px;
        border:1px solid rgba(128,128,128,.3); font-size:.8rem; margin-right:.35rem;}
.source-card {padding:.75rem; border:1px solid rgba(128,128,128,.25);
              border-radius:10px; margin:.45rem 0;}
.claim {padding:.55rem .7rem; border-left:4px solid rgba(128,128,128,.55);
        margin:.4rem 0; background:rgba(128,128,128,.06);}
</style>
""", unsafe_allow_html=True)

WORKFLOWS = {
    "Grounded RAG Chat": "Evidence-first question answering",
    "Case / Contract Review": "Risk, gaps, contradictions and missing facts",
    "Legal Drafting": "Draft from supplied facts and cited evidence",
    "Legal Research": "Research-style answer with traceable sources",
}


def _init_state() -> None:
    st.session_state.setdefault("documents", [])
    st.session_state.setdefault("chunks", [])
    st.session_state.setdefault("retriever", None)


def _process_uploads(files) -> None:
    if not files:
        return
    temp_paths: list[str] = []
    try:
        with tempfile.TemporaryDirectory() as tmp:
            for uploaded in files:
                path = Path(tmp) / uploaded.name
                path.write_bytes(uploaded.getvalue())
                temp_paths.append(str(path))
            pages = load_documents(temp_paths)
            chunks = chunk_pages(pages)
            retriever = HybridRetriever()
            retriever.index(chunks)
            configure(retriever)
            st.session_state.documents = [
                {"name": Path(p).name, "pages": sum(1 for x in pages if x["document"] == Path(p).name)}
                for p in temp_paths
            ]
            st.session_state.chunks = chunks
            st.session_state.retriever = retriever
            st.success(f"Indexed {len(st.session_state.documents)} document(s) and {len(chunks)} chunks.")
    except Exception as exc:
        st.error(f"Document processing failed: {exc}")


def _render_sources(evidence) -> None:
    st.subheader("Sources")
    if not evidence:
        st.info("No supporting evidence was returned.")
        return
    for source in evidence:
        doc = source.get("document", "Unknown")
        page = source.get("page", "—")
        chunk = source.get("chunk_id", "—")
        score = source.get("score")
        score_text = f" · score {float(score):.3f}" if isinstance(score, (int, float)) else ""
        st.markdown(
            f'<div class="source-card"><strong>{doc}</strong> — Page {page} — '
            f'<code>{chunk}</code>{score_text}<br>{source.get("text", "")}</div>',
            unsafe_allow_html=True,
        )


def _render_claims(claims) -> None:
    st.subheader("Claim verification")
    if not claims:
        st.info("No claim-level verification data was returned by the backend.")
        return
    for claim in claims:
        if isinstance(claim, str):
            status, text = "⚠", claim
        else:
            status = {"supported": "✓", "partial": "⚠", "partially_supported": "⚠",
                      "unsupported": "✗"}.get(str(claim.get("status", "")).lower(), "⚠")
            text = claim.get("claim") or claim.get("text") or str(claim)
        st.markdown(f'<div class="claim">{status} {text}</div>', unsafe_allow_html=True)


def _render_list(title: str, items, empty: str) -> None:
    st.subheader(title)
    if not items:
        st.info(empty)
        return
    for item in items:
        st.markdown(f"- {item if isinstance(item, str) else item.get('text') or item.get('claim') or item}")


def main() -> None:
    _init_state()

    with st.sidebar:
        st.markdown("## ⚖️ Agentic Legal Assistant")
        st.caption("PROVABLE · TRACEABLE · VERIFIABLE · GROUNDED")
        uploads = st.file_uploader("Upload legal documents", type=["pdf"], accept_multiple_files=True)
        if st.button("Process documents", type="primary", disabled=not uploads, use_container_width=True):
            _process_uploads(uploads)

        st.divider()
        workflow = st.selectbox("Workflow", list(WORKFLOWS))
        st.caption(WORKFLOWS[workflow])
        top_k = st.slider("Evidence sources", min_value=3, max_value=10, value=5)
        show_retrieval = st.checkbox("Show retrieval diagnostics", value=False)

        st.divider()
        st.caption("Settings")
        st.write(f"Documents loaded: **{len(st.session_state.documents)}**")
        st.write(f"Chunks indexed: **{len(st.session_state.chunks)}**")

        if st.session_state.documents:
            st.markdown("### Uploaded documents")
            for doc in st.session_state.documents:
                st.write(f"📄 {doc['name']} · {doc['pages']} pages")

    st.markdown("""
    <div class="hero">
      <h1>Legal analysis you can verify.</h1>
      <p>Every material conclusion should be traceable to a document, page and chunk.
      Unsupported claims should be visible—not hidden behind chatbot prose.</p>
      <span class="badge">PROVABLE</span><span class="badge">TRACEABLE</span>
      <span class="badge">VERIFIABLE</span><span class="badge">GROUNDED</span>
    </div>
    """, unsafe_allow_html=True)

    prompt_label = {
        "Grounded RAG Chat": "Ask a question about the provided documents",
        "Case / Contract Review": "Describe the case or contract issue to review",
        "Legal Drafting": "Describe the document or clause you want drafted",
        "Legal Research": "Ask a research question grounded in the provided materials",
    }[workflow]
    query = st.text_area(prompt_label, height=120,
                         placeholder="Example: What evidence supports the accused's arrest date?")

    if st.button("Run analysis", type="primary", use_container_width=True, disabled=not query.strip()):
        if not st.session_state.chunks:
            st.warning("Upload and process at least one document first.")
            return
        with st.spinner("Retrieving evidence and running the selected workflow…"):
            try:
                evidence = retrieve(query, top_k=top_k)
                if workflow == "Case / Contract Review":
                    raw = review_case(query)
                else:
                    context = evidence
                    raw = answer_with_grounding(query, context)
                result = normalize_answer(raw)
                result["evidence"] = result.get("evidence") or evidence
                st.session_state.last_result = result
            except Exception as exc:
                st.error(f"Analysis could not be completed: {exc}")
                return

    result = st.session_state.get("last_result")
    if not result:
        st.info("Upload documents, choose a workflow, and run an analysis to see a verifiable result.")
        return

    st.divider()
    left, right = st.columns([3, 1])
    with left:
        st.subheader("Answer")
        st.markdown(result["answer"] or "No answer was returned.")
    with right:
        confidence = result.get("confidence")
        st.metric("Confidence", str(confidence).upper() if confidence else "Not reported")

    _render_claims(result.get("claims"))
    _render_sources(result.get("evidence") or result.get("citations"))
    _render_list("Citations", result.get("citations"), "No citation records were returned.")
    _render_list("Contradictions", result.get("contradictions"), "No contradictions were reported.")
    _render_list("Missing information", result.get("missing_information"),
                 "No missing-information items were reported.")

    if show_retrieval:
        with st.expander("Retrieval diagnostics"):
            st.json(result.get("evidence") or [])

    st.caption("This interface surfaces evidence and uncertainty; it is not a substitute for professional legal advice.")


if __name__ == "__main__":
    main()
