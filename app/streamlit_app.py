"""Technical, interactive Streamlit frontend for the Agentic Legal Assistant."""
from __future__ import annotations
import sys, tempfile, html, hashlib

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
from ingestion.chunker import chunk_pages
from ingestion.pdf_loader import load_documents
from retrieval.hybrid_search import HybridRetriever
from app.backend_adapter import DRAFT_TYPES, run_workflow

st.set_page_config(
    page_title="Agentic Legal Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

WORKFLOWS = [
    ("Grounded RAG Chat", "💬", "Evidence-backed conversational analysis", "RAG"),
    ("Case / Contract Review", "🔎", "Facts, contradictions & missing data", "REVIEW"),
    ("Legal Drafting", "✍️", "Source-grounded working documents", "DRAFT"),
    ("Legal Research", "📚", "Traceable authorities & sources", "RESEARCH"),
]

st.markdown(
    """
<style>
#MainMenu, footer {visibility:hidden}
.block-container {padding:1.15rem 2.2rem 3.5rem;max-width:1550px}
[data-testid="stSidebar"] {border-right:1px solid rgba(148,163,184,.16)}
[data-testid="stSidebar"] > div {padding-top:1rem}
[data-testid="stMetric"] {background:rgba(15,23,42,.42);border:1px solid rgba(148,163,184,.14);padding:12px 14px;border-radius:14px}
.stButton>button,.stDownloadButton>button {border-radius:11px;font-weight:700}
.brand {display:flex;align-items:center;gap:12px;margin:3px 0 22px}
.logo {width:46px;height:46px;border-radius:14px;background:linear-gradient(135deg,#7c3aed,#2563eb);display:flex;align-items:center;justify-content:center;font-size:24px;box-shadow:0 8px 28px rgba(37,99,235,.22)}
.brand-title {font-size:18px;font-weight:800;line-height:1.05}
.brand-sub {font-size:10px;color:#94a3b8;margin-top:5px;letter-spacing:.4px;text-transform:uppercase}
.hero {position:relative;overflow:hidden;padding:28px 30px;border-radius:24px;background:radial-gradient(circle at 85% 15%,rgba(59,130,246,.18),transparent 34%),linear-gradient(135deg,rgba(124,58,237,.17),rgba(15,23,42,.55));border:1px solid rgba(139,92,246,.25);margin-bottom:16px}
.hero:after {content:"";position:absolute;right:-90px;bottom:-110px;width:270px;height:270px;border:1px solid rgba(96,165,250,.15);border-radius:50%;box-shadow:0 0 0 28px rgba(96,165,250,.035),0 0 0 58px rgba(96,165,250,.025)}
.hero h1 {margin:0;font-size:35px;letter-spacing:-1.2px}
.hero p {margin:8px 0 0;color:#94a3b8;font-size:14px}
.eyebrow {font-size:10px;letter-spacing:1.4px;font-weight:800;color:#a78bfa;margin-bottom:8px}
.pill {display:inline-block;padding:5px 10px;border-radius:999px;font-size:10px;font-weight:800;background:rgba(34,197,94,.11);color:#4ade80;border:1px solid rgba(34,197,94,.18);margin-bottom:10px}
.panel {padding:17px;border:1px solid rgba(148,163,184,.16);border-radius:16px;background:rgba(15,23,42,.30);margin:8px 0}
.panel-title {font-size:12px;text-transform:uppercase;letter-spacing:.9px;color:#94a3b8;font-weight:800;margin-bottom:10px}
.workflow-card {padding:14px 15px;border:1px solid rgba(148,163,184,.15);border-radius:14px;background:rgba(15,23,42,.22);min-height:100px}
.workflow-card.active {border-color:rgba(129,140,248,.5);background:linear-gradient(135deg,rgba(124,58,237,.13),rgba(37,99,235,.07))}
.workflow-icon {font-size:22px}
.workflow-code {font-family:monospace;font-size:9px;color:#818cf8;letter-spacing:1px;font-weight:800}
.workflow-name {font-weight:750;font-size:13px;margin-top:5px}
.workflow-desc {font-size:11px;color:#94a3b8;margin-top:4px;line-height:1.45}
.pipeline {display:flex;gap:6px;align-items:center;flex-wrap:wrap;margin:6px 0 15px}
.node {padding:7px 10px;border:1px solid rgba(148,163,184,.16);border-radius:9px;background:rgba(15,23,42,.34);font-size:10px;color:#cbd5e1}
.node.active {border-color:rgba(96,165,250,.45);color:#93c5fd;background:rgba(37,99,235,.08)}
.arrow {color:#64748b;font-size:12px}
.source {padding:15px 16px;border:1px solid rgba(148,163,184,.15);border-radius:14px;margin:9px 0;background:linear-gradient(135deg,rgba(15,23,42,.38),rgba(15,23,42,.18))}
.source:hover {border-color:rgba(96,165,250,.35)}
.source-title {font-weight:750;font-size:13px}
.source-meta {font-size:10px;color:#94a3b8;margin:4px 0 9px;font-family:monospace}
.source-text {font-size:12px;line-height:1.6;color:#dbe4f0}
.badge {display:inline-block;padding:4px 8px;border-radius:999px;font-size:9px;font-weight:850;letter-spacing:.3px;margin-right:6px}
.good {background:rgba(34,197,94,.12);color:#4ade80}
.warn {background:rgba(245,158,11,.12);color:#fbbf24}
.bad {background:rgba(239,68,68,.12);color:#f87171}
.neutral {background:rgba(148,163,184,.11);color:#cbd5e1}
.trace {font-family:monospace;font-size:10px;color:#94a3b8;line-height:1.7}
.empty {padding:55px 25px;text-align:center;border:1px dashed rgba(148,163,184,.23);border-radius:20px;color:#94a3b8;background:rgba(15,23,42,.15)}
.small {font-size:11px;color:#94a3b8}
.chat-q {padding:11px 13px;border-left:2px solid #6366f1;background:rgba(99,102,241,.06);font-size:12px;color:#cbd5e1;border-radius:0 10px 10px 0;margin:8px 0}
</style>
""",
    unsafe_allow_html=True,
)


def init():
    defaults = {
        "documents": [],
        "chunks": [],
        "last_result": None,
        "chat_history": [],
        "last_query": "",
        "processed": False,
        "last_workflow": "Grounded RAG Chat",
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


def process(files):
    with tempfile.TemporaryDirectory() as tmp:
        paths = []
        existing_names = set()
        for f in files:
            name = Path(f.name).name
            if name != f.name or name in existing_names:
                raise ValueError("Upload PDFs with unique plain filenames.")
            existing_names.add(name)
            p = Path(tmp) / name
            p.write_bytes(f.getvalue())
            paths.append(str(p))

        pages = load_documents(paths)
        chunks = chunk_pages(pages)
        if not chunks:
            raise ValueError("No readable text found. Upload a text PDF or enable OCR.")

        manifest = b"".join(
            f.name.encode("utf-8") + b"\0" + f.getvalue() + b"\0" for f in files
        )
        namespace = hashlib.sha256(manifest).hexdigest()[:16]
        r = HybridRetriever(namespace=namespace)
        r.index(chunks)

        st.session_state.retriever = r
        st.session_state.documents = [
            {
                "name": Path(p).name,
                "pages": sum(x["document"] == Path(p).name for x in pages),
            }
            for p in paths
        ]
        st.session_state.chunks = chunks
        st.session_state.chat_history = []
        st.session_state.last_result = None
        st.session_state.last_query = ""
        st.session_state.processed = True


def badge(status):
    s = str(status or "UNKNOWN").upper()
    cls = (
        "good" if s in {"SUPPORTED", "VERIFIED", "HIGH", "READY_FOR_REVIEW"}
        else "bad" if s in {"CONTRADICTED", "UNSUPPORTED", "LOW"}
        else "warn" if s in {"PARTIALLY_SUPPORTED", "UNCERTAIN", "INCOMPLETE"}
        else "neutral"
    )
    return f'<span class="badge {cls}">{html.escape(s)}</span>'


def render_pipeline(workflow, stage="Evidence verification"):
    labels = ["Ingest", "Hybrid retrieval", "Rerank", "Reason", "Verify", "Cite"]
    st.markdown('<div class="panel"><div class="panel-title">Analysis pipeline</div><div class="pipeline">', unsafe_allow_html=True)
    parts = []
    for i, label in enumerate(labels):
        active = " active" if (stage == "Evidence verification" and i >= 1) or (stage == "Generation" and i == 3) else ""
        parts.append(f'<span class="node{active}">{i+1:02d} · {label}</span>')
        if i < len(labels)-1:
            parts.append('<span class="arrow">→</span>')
    st.markdown("".join(parts) + "</div></div>", unsafe_allow_html=True)


def sources(items):
    st.markdown("### Evidence")
    if not items:
        st.info("No evidence was returned for this request.")
        return

    for i, x in enumerate(items):
        if not isinstance(x, dict):
            continue
        txt = x.get("text") or x.get("relevant_passage") or x.get("statement") or ""
        src = x.get("source") if isinstance(x.get("source"), dict) else x
        doc = html.escape(str(src.get("document", "Unknown")))
        page = html.escape(str(src.get("page", "—")))
        chunk = html.escape(str(src.get("chunk_id", "—")))
        score = src.get("score", x.get("score"))
        score_text = f" · score {float(score):.3f}" if isinstance(score, (int, float)) else ""
        with st.expander(f"#{i+1}  📄 {doc}  ·  page {page}", expanded=i == 0):
            c1, c2 = st.columns([5, 1])
            with c1:
                st.markdown(f'<div class="source-text">{html.escape(str(txt))}</div>', unsafe_allow_html=True)
            with c2:
                st.markdown(f'<div class="trace">PAGE<br>{page}<br><br>CHUNK<br>{chunk}<br>{html.escape(score_text)}</div>', unsafe_allow_html=True)


def claims(items):
    st.markdown("### Claim verification")
    if not items:
        st.info("No claim-level verification was returned.")
        return
    for i, x in enumerate(items):
        if isinstance(x, str):
            st.markdown(f"- {html.escape(x)}")
            continue
        status = x.get("status", "UNKNOWN")
        claim = x.get("claim_text") or x.get("claim") or x.get("text", "")
        st.markdown(
            f'<div class="panel"><span class="trace">CLAIM {i+1:02d}</span><br>{badge(status)} {html.escape(str(claim))}</div>',
            unsafe_allow_html=True,
        )


def citations(items):
    st.markdown("### Verified citations")
    if not items:
        st.info("No verified citations returned.")
        return
    for x in items:
        if not isinstance(x, dict):
            continue
        valid = x.get("valid")
        doc = x.get("document", "Unknown")
        page = x.get("page", "—")
        chunk = x.get("chunk_id", "—")
        claim = x.get("claim_text", "")
        st.markdown(
            f'<div class="panel">{badge("VERIFIED" if valid else "CHECK")} <b>{html.escape(str(doc))}</b> · Page {html.escape(str(page))}<br><span class="small">{html.escape(str(claim))}</span><br><span class="trace">chunk={html.escape(str(chunk))}</span></div>',
            unsafe_allow_html=True,
        )


def findings(items):
    for x in items or []:
        if isinstance(x, str):
            st.markdown(f"- {html.escape(x)}")
        elif isinstance(x, dict):
            label = x.get("status") or x.get("type") or x.get("message") or x.get("claim_text")
            detail = x.get("message") or x.get("description") or x.get("detail")
            st.markdown(
                f'<div class="panel">{badge(label)} {html.escape(str(detail or ""))}</div>',
                unsafe_allow_html=True,
            )


def render_result(r, workflow):
    # Blocked/off-topic requests get a clean redirect UI. No retrieval,
    # confidence, claim or citation metrics are shown because analysis stopped.
    if r.get("guard_blocked"):
        st.divider()
        st.warning("⚠️ This question is outside the scope of the uploaded legal case.")
        st.markdown(f'<div class="panel" style="font-size:15px;line-height:1.7">{html.escape(str(r.get("answer", "")))}</div>', unsafe_allow_html=True)
        suggestions = r.get("suggestions") or []
        if suggestions:
            st.markdown("### Suggested case questions")
            cols = st.columns(min(3, len(suggestions)))
            for i, suggestion in enumerate(suggestions):
                with cols[i % len(cols)]:
                    st.info(str(suggestion))
        return

    st.divider()
    confidence = r.get("confidence")
    evidence = r.get("evidence") or r.get("sources") or []
    verified = sum(
        x.get("status") == "SUPPORTED"
        for x in (r.get("claims") or [])
        if isinstance(x, dict)
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Evidence", len(evidence))
    c2.metric("Confidence", str(confidence).upper() if confidence else "—")
    c3.metric("Verified claims", verified)
    c4.metric("Traceable citations", sum(bool(x.get("valid")) for x in (r.get("citations") or []) if isinstance(x, dict)))

    if workflow == "Grounded RAG Chat":
        provider = {"gemini": "Gemini", "openai": "OpenAI"}.get(r.get("llm_provider"), "LLM")
        if r.get("llm_used"):
            st.success(f"🤖 {provider} reasoning active · final response passed deterministic grounding checks.")
        elif r.get("llm_enabled"):
            st.warning(f"{provider} is configured, but this response used the safe deterministic fallback.")
        else:
            st.info("Deterministic grounding mode is active.")

    if r.get("answer"):
        if r.get("guard_blocked"):
            st.warning("⚠️ This question is outside the scope of the uploaded legal case.")
            st.markdown(f'<div class="panel" style="font-size:15px;line-height:1.7">{html.escape(str(r["answer"]))}</div>', unsafe_allow_html=True)
            suggestions = r.get("suggestions") or []
            if suggestions:
                st.markdown("### Suggested case questions")
                cols = st.columns(min(3, len(suggestions)))
                for i, suggestion in enumerate(suggestions):
                    with cols[i % len(cols)]:
                        st.info(str(suggestion))
            return

        st.markdown("### Answer")
        st.markdown(f'<div class="panel" style="font-size:16px;line-height:1.75">{html.escape(str(r["answer"]))}</div>', unsafe_allow_html=True)

    if r.get("key_facts"):
        st.markdown("### Key facts")
        st.dataframe(r["key_facts"], use_container_width=True, hide_index=True)

    if r.get("authorities"):
        st.markdown("### Research authorities")
        for a in r["authorities"]:
            title = a.get("case_name") or a.get("citation") or "Authority"
            st.markdown(
                f'<div class="panel"><b>📚 {html.escape(str(title))}</b><br><span class="small">{html.escape(str(a.get("court","")))} · {html.escape(str(a.get("date_year","")))}</span><br>{html.escape(str(a.get("relevant_passage","")))}</div>',
                unsafe_allow_html=True,
            )

    if r.get("draft"):
        st.markdown("### Grounded working draft")
        st.code(r["draft"], language="text")
        st.caption("Evidence-grounded working draft — review placeholders and citations before use.")

    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Evidence", "🧠 Claims", "⚠️ Issues", "🔗 Traceability"])
    with tab1:
        sources(evidence)
    with tab2:
        claims(r.get("claims") or r.get("claim_verification"))
        citations(r.get("citations"))
    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Contradictions")
            findings(r.get("contradictions") or [])
        with col2:
            st.markdown("#### Missing information")
            findings(r.get("missing_information") or [])
        st.markdown("#### Warnings")
        findings(r.get("warnings") or [])
    with tab4:
        st.markdown("#### Evidence trace")
        st.caption("A claim is considered safe only when its supporting evidence and citation linkage pass verification.")
        citations(r.get("citations"))

        with st.expander("Technical response metadata"):
            safe_meta = {
                "workflow": workflow,
                "confidence": r.get("confidence"),
                "evidence_count": len(evidence),
                "claim_count": len(r.get("claims") or []),
                "citation_count": len(r.get("citations") or []),
                "llm_used": r.get("llm_used"),
            }
            st.json(safe_meta)


def render_workspace_overview(workflow):
    st.markdown("### System workspace")
    cols = st.columns(4)
    for col, (name, icon, desc, code) in zip(cols, WORKFLOWS):
        active = name == workflow
        with col:
            st.markdown(
                f'<div class="workflow-card {"active" if active else ""}"><div class="workflow-icon">{icon}</div><div class="workflow-code">{code}</div><div class="workflow-name">{html.escape(name)}</div><div class="workflow-desc">{html.escape(desc)}</div></div>',
                unsafe_allow_html=True,
            )
    render_pipeline(workflow)


def main():
    init()

    with st.sidebar:
        st.markdown(
            '<div class="brand"><div class="logo">⚖️</div><div><div class="brand-title">Agentic Legal AI</div><div class="brand-sub">Evidence-first legal intelligence</div></div></div>',
            unsafe_allow_html=True,
        )

        files = st.file_uploader(
            "📄 Upload legal PDFs",
            type=["pdf"],
            accept_multiple_files=True,
            help="Upload the case bundle, contract, notices, exhibits or supporting documents.",
        )
        if files:
            st.caption(f"READY · {len(files)} document(s) selected")

        if st.button(
            "⚡ Process & Index Evidence",
            type="primary",
            disabled=not files,
            use_container_width=True,
        ):
            try:
                with st.spinner("Extracting → chunking → indexing evidence…"):
                    process(files)
                st.success(f"Indexed {len(st.session_state.chunks)} evidence chunks.")
            except Exception as e:
                st.error(f"Processing failed: {e}")

        st.markdown("### Workspace")
        workflow_names = [x[0] for x in WORKFLOWS]
        workflow = st.radio(
            "Select analysis mode",
            workflow_names,
            index=workflow_names.index(st.session_state.get("last_workflow", workflow_names[0])),
            label_visibility="collapsed",
        )
        document_type = "legal notice"
        if workflow == "Legal Drafting":
            document_type = st.selectbox("Draft type", list(DRAFT_TYPES), format_func=str.title)

        st.markdown("### Retrieval controls")
        top_k = st.slider(
            "Evidence sources",
            3,
            10,
            5,
            help="Number of top-ranked evidence chunks passed to the grounding layer.",
        )

        st.divider()
        c1, c2 = st.columns(2)
        c1.metric("Docs", len(st.session_state.documents))
        c2.metric("Chunks", len(st.session_state.chunks))
        for d in st.session_state.documents:
            st.caption(f"📄 {d['name']} · {d['pages']} pages")

        if workflow == "Grounded RAG Chat" and st.button("Clear conversation", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.last_result = None
            st.session_state.last_query = ""

    st.markdown(
        '<div class="hero"><div class="eyebrow">AGENTIC LEGAL ASSISTANT · EVIDENCE ENGINE</div><span class="pill">GROUNDED · TRACEABLE · VERIFIABLE</span><h1>Legal analysis you can inspect.</h1><p>Retrieve evidence → rank sources → reason → verify claims → validate citations → respond.</p></div>',
        unsafe_allow_html=True,
    )

    if not st.session_state.chunks:
        st.markdown(
            '<div class="empty"><div style="font-size:38px">⚖️</div><h3>Initialize the legal evidence workspace</h3><p>Upload your case bundle or contract PDFs from the sidebar. The system will parse, chunk and index the evidence before analysis.</p></div>',
            unsafe_allow_html=True,
        )
        return

    render_workspace_overview(workflow)

    if workflow == "Grounded RAG Chat":
        st.markdown("### 💬 Grounded RAG Chat")
        st.caption("Conversational Q&A over the indexed case bundle. Answers remain constrained by retrieved evidence.")
        for m in st.session_state.chat_history:
            with st.chat_message(m["role"]):
                st.write(m["content"])
        q = st.chat_input("Ask about facts, evidence, clauses, dates, contradictions or legal issues…")
        run = True
    else:
        descriptions = {
            "Case / Contract Review": "Identify key facts, contradictions, missing information and evidence-backed issues.",
            "Legal Drafting": "Generate a working draft using only supported case facts and traceable evidence.",
            "Legal Research": "Explore indexed legal authorities and preserve source provenance.",
        }
        st.markdown(f"### {dict((x[0], x[1]) for x in WORKFLOWS)[workflow]} {workflow}")
        st.caption(descriptions[workflow])

        prompts = {
            "Case / Contract Review": "Example: Identify the delivery obligation, deadline, breach evidence and contradictions.",
            "Legal Drafting": "Example: Draft a legal notice based only on the uploaded evidence.",
            "Legal Research": "Example: What indexed authority is relevant to the disputed contractual obligation?",
        }
        q = st.text_area(
            "Analysis request",
            height=120,
            placeholder=prompts[workflow],
            label_visibility="collapsed",
        )
        run = st.button(
            "🚀 Run evidence-grounded analysis",
            type="primary",
            use_container_width=True,
            disabled=not q.strip(),
        )

    if q and st.session_state.chunks:
        if workflow != "Grounded RAG Chat" and not run:
            return
        try:
            with st.spinner("Retrieving → reasoning → verifying → validating citations…"):
                prior = st.session_state.chat_history if workflow == "Grounded RAG Chat" else []
                result = run_workflow(
                    workflow,
                    q,
                    top_k,
                    document_type,
                    conversation_context=prior,
                    retriever=st.session_state.retriever,
                )
            st.session_state.last_result = result
            st.session_state.last_workflow = workflow
            st.session_state.last_query = q
            if workflow == "Grounded RAG Chat":
                st.session_state.chat_history += [
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": result.get("answer", "")},
                ]
        except Exception as e:
            st.error(f"Analysis failed: {e}")
            return

    r = st.session_state.last_result
    if not r:
        st.markdown(
            '<div class="empty"><div style="font-size:30px">🔬</div><h3>Evidence workspace ready</h3><p>Run an analysis to inspect retrieval, claim verification, contradictions, missing information and citation traceability.</p></div>',
            unsafe_allow_html=True,
        )
        return

    if st.session_state.get("last_workflow") == workflow:
        render_result(r, workflow)

    st.caption("⚖️ Evidence-grounded prototype · Not legal advice")


if __name__ == "__main__":
    main()
