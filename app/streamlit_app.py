"""Advanced Streamlit frontend for the integrated Agentic Legal Assistant."""
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

st.set_page_config(page_title="Agentic Legal Assistant", page_icon="⚖️", layout="wide", initial_sidebar_state="expanded")

WORKFLOWS = [
    ("Grounded RAG Chat", "💬", "Ask questions and get evidence-backed answers"),
    ("Case / Contract Review", "🔎", "Facts, contradictions and missing information"),
    ("Legal Drafting", "✍️", "Create an evidence-grounded working draft"),
    ("Legal Research", "📚", "Explore indexed authorities and sources"),
]

st.markdown("""
<style>
#MainMenu, footer {visibility:hidden}
.block-container {padding: 1.2rem 2.2rem 3rem; max-width: 1500px}
[data-testid="stSidebar"] {border-right:1px solid rgba(148,163,184,.18)}
[data-testid="stSidebar"] > div {padding-top:1.2rem}
.brand {display:flex;align-items:center;gap:12px;margin-bottom:18px}
.logo {width:44px;height:44px;border-radius:13px;background:linear-gradient(135deg,#7c3aed,#2563eb);display:flex;align-items:center;justify-content:center;font-size:23px}
.brand-title {font-size:18px;font-weight:750;line-height:1.1}
.brand-sub {font-size:11px;color:#94a3b8;margin-top:3px}
.hero {padding:28px 30px;border-radius:22px;background:linear-gradient(135deg,rgba(124,58,237,.16),rgba(37,99,235,.10));border:1px solid rgba(139,92,246,.25);margin-bottom:18px}
.hero h1 {margin:0;font-size:34px;letter-spacing:-1px}
.hero p {margin:8px 0 0;color:#94a3b8;font-size:15px}
.pill {display:inline-block;padding:5px 10px;border-radius:999px;font-size:11px;font-weight:700;background:rgba(34,197,94,.12);color:#4ade80;border:1px solid rgba(34,197,94,.2);margin-bottom:10px}
.card {padding:18px;border:1px solid rgba(148,163,184,.18);border-radius:16px;background:rgba(15,23,42,.28);margin:8px 0}
.source {padding:14px 16px;border:1px solid rgba(148,163,184,.16);border-radius:14px;margin:8px 0;background:rgba(15,23,42,.20)}
.source-title {font-weight:700;font-size:14px}
.source-meta {font-size:11px;color:#94a3b8;margin:3px 0 8px}
.source-text {font-size:13px;line-height:1.55}
.badge {display:inline-block;padding:4px 8px;border-radius:999px;font-size:10px;font-weight:800;margin-right:6px}
.good {background:rgba(34,197,94,.13);color:#4ade80}
.warn {background:rgba(245,158,11,.13);color:#fbbf24}
.bad {background:rgba(239,68,68,.13);color:#f87171}
.neutral {background:rgba(148,163,184,.13);color:#cbd5e1}
.workflow-card {padding:13px 14px;border:1px solid rgba(148,163,184,.14);border-radius:13px;margin:7px 0}
.empty {padding:45px 20px;text-align:center;border:1px dashed rgba(148,163,184,.25);border-radius:18px;color:#94a3b8}
.small {font-size:12px;color:#94a3b8}
</style>
""", unsafe_allow_html=True)

def init():
    defaults = {"documents":[],"chunks":[],"last_result":None,"chat_history":[],"last_query":"","processed":False}
    for k,v in defaults.items():
        st.session_state.setdefault(k,v)

def process(files):
    with tempfile.TemporaryDirectory() as tmp:
        paths=[]
        for f in files:
            name = Path(f.name).name
            if name != f.name or name in [Path(x).name for x in paths]:
                raise ValueError("Upload PDFs with unique plain filenames.")
            p=Path(tmp)/name
            p.write_bytes(f.getvalue())
            paths.append(str(p))
        pages=load_documents(paths)
        chunks=chunk_pages(pages)
        if not chunks:
            raise ValueError("No readable text found. Upload a text PDF or enable OCR.")
        # Namespace the persistent vector collection by the exact uploaded
        # document set. A changed file or changed filename gets a new namespace,
        # preventing stale chunks from previous uploads from leaking into results.
        manifest = b"".join(
            f.name.encode("utf-8") + b"\\0" + f.getvalue() + b"\\0"
            for f in files
        )
        namespace = hashlib.sha256(manifest).hexdigest()[:16]
        r=HybridRetriever(namespace=namespace)
        r.index(chunks)
        st.session_state.retriever=r
        st.session_state.documents=[{"name":Path(p).name,"pages":sum(x["document"]==Path(p).name for x in pages)} for p in paths]
        st.session_state.chunks=chunks
        st.session_state.chat_history=[]
        st.session_state.last_result=None
        st.session_state.last_query=""
        st.session_state.processed=True

def badge(status):
    s=str(status or "UNKNOWN").upper()
    cls="good" if s in {"SUPPORTED","VERIFIED","HIGH","READY_FOR_REVIEW"} else "bad" if s in {"CONTRADICTED","UNSUPPORTED","LOW"} else "warn" if s in {"PARTIALLY_SUPPORTED","UNCERTAIN","INCOMPLETE"} else "neutral"
    return f'<span class="badge {cls}">{html.escape(s)}</span>'

def sources(items):
    st.markdown("### Evidence")
    if not items:
        st.info("No evidence was returned for this request.")
        return
    for x in items:
        if not isinstance(x,dict): continue
        txt=x.get("text") or x.get("relevant_passage") or x.get("statement") or ""
        src=x.get("source") if isinstance(x.get("source"),dict) else x
        doc=html.escape(str(src.get("document","Unknown")))
        page=html.escape(str(src.get("page","—")))
        chunk=html.escape(str(src.get("chunk_id","—")))
        st.markdown(f'<div class="source"><div class="source-title">📄 {doc}</div><div class="source-meta">Page {page} · Evidence chunk {chunk}</div><div class="source-text">{html.escape(str(txt))}</div></div>',unsafe_allow_html=True)

def claims(items):
    st.markdown("### Claim verification")
    if not items:
        st.info("No claim-level verification was returned.")
        return
    for x in items:
        if isinstance(x,str):
            st.markdown(f"- {x}")
            continue
        status=x.get("status","UNKNOWN")
        claim=x.get("claim_text") or x.get("claim") or x.get("text","")
        st.markdown(f'<div class="card">{badge(status)} {html.escape(str(claim))}</div>',unsafe_allow_html=True)

def citations(items):
    st.markdown("### Verified citations")
    if not items:
        st.info("No verified citations returned.")
        return
    for x in items:
        if not isinstance(x,dict): continue
        valid=x.get("valid")
        doc=x.get("document","Unknown"); page=x.get("page","—"); chunk=x.get("chunk_id","—")
        claim=x.get("claim_text","")
        st.markdown(f'<div class="card">{badge("VERIFIED" if valid else "CHECK")} <b>{html.escape(str(doc))}</b> · Page {html.escape(str(page))}<br><span class="small">{html.escape(str(claim))}</span><br><span class="small">Chunk: {html.escape(str(chunk))}</span></div>',unsafe_allow_html=True)

def findings(items):
    for x in items or []:
        if isinstance(x,str):
            st.markdown(f"- {x}")
        elif isinstance(x,dict):
            label=x.get("status") or x.get("type") or x.get("message") or x.get("claim_text")
            detail=x.get("message") or x.get("description")
            st.markdown(f'<div class="card">{badge(label)} {html.escape(str(detail or ""))}</div>',unsafe_allow_html=True)

def render_result(r, workflow):
    st.divider()
    route=r.get("route","workflow-specific")
    confidence=r.get("confidence")
    c1,c2,c3=st.columns([2,2,2])
    with c1:
        st.metric("Evidence sources",len(r.get("evidence") or r.get("sources") or []))
    with c2:
        st.metric("Confidence",str(confidence).upper() if confidence else "—")
    with c3:
        st.metric("Verified claims",sum(x.get("status")=="SUPPORTED" for x in (r.get("claims") or []) if isinstance(x,dict)))
    # Internal routing is intentionally hidden from users. The UI should\n    # show the actual result, not implementation-level route names such as\n    # CASE_RELEVANCE_GUARD. Guarded/off-topic responses are explained below.\n    if r.get("guard_blocked"):\n        pass

    if workflow=="Grounded RAG Chat":
        provider = {"gemini": "Gemini", "openai": "OpenAI"}.get(r.get("llm_provider"), "LLM")
        if r.get("llm_used"):
            st.success(f"🤖 {provider} response generated · claims checked against retrieved evidence.")
        elif r.get("llm_enabled"):
            st.warning(f"{provider} is configured, but this response used the deterministic fallback. Check the model, API key and quota.")
        else:
            st.warning("Deterministic fallback is active. Configure Gemini or OpenAI to enable LLM responses.")

    if r.get("answer"):
        if r.get("guard_blocked"):
            st.warning("⚠️ Inappropriate / off-topic question")
            st.markdown(f'<div class="card" style="font-size:16px;line-height:1.7">{html.escape(str(r["answer"]))}</div>',unsafe_allow_html=True)
            suggestions = r.get("suggestions") or []
            if suggestions:
                st.markdown("### Ask something related to this case")
                for suggestion in suggestions:
                    st.markdown(f"- {html.escape(str(suggestion))}")
            return
        st.markdown("### Answer")
        st.markdown(f'<div class="card" style="font-size:16px;line-height:1.7">{html.escape(str(r["answer"]))}</div>',unsafe_allow_html=True)

    if r.get("key_facts"):
        st.markdown("### Key facts")
        st.dataframe(r["key_facts"],use_container_width=True,hide_index=True)

    if r.get("authorities"):
        st.markdown("### Research authorities")
        for a in r["authorities"]:
            title=a.get("case_name") or a.get("citation") or "Authority"
            st.markdown(f'<div class="card"><b>📚 {html.escape(str(title))}</b><br><span class="small">{html.escape(str(a.get("court",""))) } · {html.escape(str(a.get("date_year","")))}</span><br>{html.escape(str(a.get("relevant_passage","")))}</div>',unsafe_allow_html=True)

    if r.get("draft"):
        st.markdown("### Grounded working draft")
        st.code(r["draft"],language="text")
        st.caption("Evidence-grounded working draft — review placeholders and citations before use.")

    tab1,tab2,tab3,tab4=st.tabs(["🔍 Evidence","✅ Verification","⚠️ Issues","📌 Sources"])
    with tab1:
        sources(r.get("evidence") or r.get("sources") or r.get("authorities"))
    with tab2:
        claims(r.get("claims") or r.get("claim_verification"))
        citations(r.get("citations"))
    with tab3:
        st.markdown("#### Contradictions")
        findings(r.get("contradictions") or [])
        st.markdown("#### Missing information")
        findings(r.get("missing_information") or [])
        st.markdown("#### Warnings")
        findings(r.get("warnings") or [])
    with tab4:
        st.markdown("#### Traceability")
        st.caption("Every supported claim should be traceable to the evidence shown above.")
        citations(r.get("citations"))

def main():
    init()
    with st.sidebar:
        st.markdown('<div class="brand"><div class="logo">⚖️</div><div><div class="brand-title">Agentic Legal AI</div><div class="brand-sub">Evidence-first legal assistant</div></div></div>',unsafe_allow_html=True)
        files=st.file_uploader("📄 Upload legal PDFs",type=["pdf"],accept_multiple_files=True)
        if files:
            st.caption(f"{len(files)} document(s) selected")
        if st.button("⚡ Process & Index Documents",type="primary",disabled=not files,use_container_width=True):
            try:
                with st.spinner("Extracting, chunking and indexing evidence…"):
                    process(files)
                st.success(f"Indexed {len(st.session_state.chunks)} evidence chunks.")
            except Exception as e:
                st.error(f"Processing failed: {e}")

        st.markdown("### Workspace")
        workflow_names=[x[0] for x in WORKFLOWS]
        workflow=st.radio("Choose workflow",workflow_names,index=0,label_visibility="collapsed")
        document_type="legal notice"
        if workflow=="Legal Drafting":
            document_type=st.selectbox("Document type",list(DRAFT_TYPES),format_func=str.title)

        st.markdown("### Retrieval")
        top_k=st.slider("Evidence sources",3,10,5)
        st.divider()
        d1,d2=st.columns(2)
        d1.metric("Documents",len(st.session_state.documents))
        d2.metric("Chunks",len(st.session_state.chunks))
        for d in st.session_state.documents:
            st.caption(f"📄 {d['name']} · {d['pages']} pages")
        if workflow=="Grounded RAG Chat" and st.button("Clear conversation",use_container_width=True):
            st.session_state.chat_history=[]
            st.session_state.last_result=None
            st.session_state.last_query=""

    st.markdown('<div class="hero"><span class="pill">EVIDENCE FIRST · GROUNDED · TRACEABLE</span><h1>Legal analysis you can verify.</h1><p>Retrieve evidence → reason → verify claims → validate citations → answer.</p></div>',unsafe_allow_html=True)

    if not st.session_state.chunks:
        st.markdown('<div class="empty"><div style="font-size:34px">📑</div><h3>Start by uploading legal documents</h3><p>Process your PDFs from the sidebar. Processing only indexes evidence — answers are generated after you ask a question.</p></div>',unsafe_allow_html=True)
        return

    if workflow=="Grounded RAG Chat":
        st.markdown("### 💬 Grounded RAG Chat")
        st.caption("Ask questions about the uploaded documents. Follow-up questions can use the conversation context.")
        for m in st.session_state.chat_history:
            with st.chat_message(m["role"]):
                st.write(m["content"])
        q=st.chat_input("Ask a question about the uploaded legal documents…")
        run=True
    else:
        descriptions={
            "Case / Contract Review":"Identify key facts, contradictions, and missing information from the evidence.",
            "Legal Drafting":"Generate an evidence-grounded working draft without inventing facts.",
            "Legal Research":"Find and summarize relevant indexed legal authorities with traceable evidence.",
        }
        st.markdown(f"### {dict((x[0],x[1]) for x in WORKFLOWS)[workflow]} {workflow}")
        st.caption(descriptions[workflow])
        prompts={"Case / Contract Review":"Describe the case or contract issue…","Legal Drafting":"Describe what you need drafted…","Legal Research":"Ask your legal research question…"}
        q=st.text_area("Your request",height=120,placeholder=prompts[workflow],label_visibility="collapsed")
        run=st.button("🚀 Run grounded analysis",type="primary",use_container_width=True,disabled=not q.strip())

    if q and st.session_state.chunks:
        if workflow!="Grounded RAG Chat" and not run:
            return
        try:
            with st.spinner("Retrieving evidence and verifying the response…"):
                prior=st.session_state.chat_history if workflow=="Grounded RAG Chat" else []
                result=run_workflow(workflow,q,top_k,document_type,conversation_context=prior, retriever=st.session_state.retriever)
            st.session_state.last_result=result
            st.session_state.last_workflow=workflow
            st.session_state.last_query=q
            if workflow=="Grounded RAG Chat":
                st.session_state.chat_history += [{"role":"user","content":q},{"role":"assistant","content":result.get("answer","")}]
        except Exception as e:
            st.error(f"Analysis failed: {e}")
            return

    r=st.session_state.last_result
    if not r:
        st.markdown('<div class="empty"><div style="font-size:30px">🔎</div><h3>Your evidence workspace is ready</h3><p>Enter a question above to retrieve evidence and generate a verified response.</p></div>',unsafe_allow_html=True)
        return

    if st.session_state.get("last_workflow") == workflow:
        render_result(r,workflow)
    st.caption("⚖️ Evidence-grounded prototype · Not legal advice")

if __name__=="__main__":
    main()
