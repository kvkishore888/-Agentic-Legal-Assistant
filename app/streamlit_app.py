"""Streamlit frontend for the integrated Agentic Legal Assistant."""
from __future__ import annotations
import sys, tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
import streamlit as st
from ingestion.chunker import chunk_pages
from ingestion.pdf_loader import load_documents
from retrieval.hybrid_search import HybridRetriever, configure
from app.backend_adapter import DRAFT_TYPES, run_workflow

st.set_page_config(page_title="Agentic Legal Assistant", page_icon="⚖️", layout="wide")
st.markdown("""<style>.block-container{padding-top:1.5rem}.hero{padding:1.2rem;border:1px solid rgba(128,128,128,.25);border-radius:16px;margin-bottom:1rem}.source{padding:.7rem;border:1px solid rgba(128,128,128,.2);border-radius:10px;margin:.4rem 0}</style>""",unsafe_allow_html=True)
WORKFLOWS=["Grounded RAG Chat","Case / Contract Review","Legal Drafting","Legal Research"]

def init():
    for k,v in {"documents":[],"chunks":[],"last_result":None,"chat_history":[]}.items(): st.session_state.setdefault(k,v)

def process(files):
    with tempfile.TemporaryDirectory() as tmp:
        paths=[]
        for f in files:
            p=Path(tmp)/f.name; p.write_bytes(f.getvalue()); paths.append(str(p))
        pages=load_documents(paths); chunks=chunk_pages(pages); r=HybridRetriever(); r.index(chunks); configure(r)
        st.session_state.documents=[{"name":Path(p).name,"pages":sum(x["document"]==Path(p).name for x in pages)} for p in paths]
        st.session_state.chunks=chunks; st.session_state.chat_history=[]; st.session_state.last_result=None

def sources(items):
    st.subheader("Evidence & Sources")
    for x in items or []:
        if not isinstance(x,dict): continue
        txt=x.get("text") or x.get("relevant_passage") or x.get("statement") or ""
        src=x.get("source") if isinstance(x.get("source"),dict) else x
        st.markdown(f'<div class="source"><b>{src.get("document","Unknown")}</b> · Page {src.get("page","—")} · <code>{src.get("chunk_id","—")}</code><br>{txt}</div>',unsafe_allow_html=True)

def claims(items):
    st.subheader("Claim Verification")
    if not items: st.info("No claim-level verification was returned."); return
    for x in items:
        if isinstance(x,str): st.write(x)
        else: st.write(f'**{x.get("status","UNKNOWN")}** — {x.get("claim_text") or x.get("claim") or x.get("text","")}')

def main():
    init()
    with st.sidebar:
        st.markdown("## ⚖️ Agentic Legal Assistant")
        files=st.file_uploader("Upload PDF legal documents",type=["pdf"],accept_multiple_files=True)
        if st.button("Process documents",type="primary",disabled=not files,use_container_width=True):
            try: process(files); st.success(f"Indexed {len(st.session_state.chunks)} chunks.")
            except Exception as e: st.error(f"Processing failed: {e}")
        workflow=st.selectbox("Workflow",WORKFLOWS)
        document_type="legal notice"
        if workflow=="Legal Drafting": document_type=st.selectbox("Draft type",list(DRAFT_TYPES),format_func=str.title)
        top_k=st.slider("Evidence sources",3,10,5)
        st.divider(); st.write(f"Documents: **{len(st.session_state.documents)}**"); st.write(f"Chunks: **{len(st.session_state.chunks)}**")
        for d in st.session_state.documents: st.caption(f"📄 {d['name']} · {d['pages']} pages")
        if workflow=="Grounded RAG Chat" and st.button("Clear chat"):
            st.session_state.chat_history=[]; st.session_state.last_result=None

    st.markdown('<div class="hero"><h1>Legal analysis you can verify.</h1><p>Evidence first: retrieve → reason → verify → cite. Unsupported claims stay visible.</p></div>',unsafe_allow_html=True)
    if workflow=="Grounded RAG Chat":
        for m in st.session_state.chat_history: st.chat_message(m["role"]).write(m["content"])
        q=st.chat_input("Ask a question about the uploaded documents")
    else:
        prompts={"Case / Contract Review":"Describe the case or contract issue","Legal Drafting":"Describe the document you need drafted","Legal Research":"Ask a legal research question"}
        q=st.text_area(prompts[workflow],height=110)
        run=st.button("Run analysis",type="primary",use_container_width=True,disabled=not q.strip())
    if q and st.session_state.chunks:
        if workflow!="Grounded RAG Chat" and not run: return
        try:
            with st.spinner("Running integrated workflow…"):
                prior=st.session_state.chat_history if workflow=="Grounded RAG Chat" else []
                result=run_workflow(workflow,q,top_k,document_type,conversation_context=prior)
                st.session_state.last_result=result
                if workflow=="Grounded RAG Chat":
                    st.session_state.chat_history += [{"role":"user","content":q},{"role":"assistant","content":result.get("answer","")}]
        except Exception as e: st.error(f"Analysis failed: {e}"); return
    elif q and not st.session_state.chunks:
        st.warning("Process at least one document first."); return

    r=st.session_state.last_result
    if not r: st.info("Upload documents, choose a workflow, and run an analysis."); return
    st.divider(); st.caption(f"Route: {r.get('route','workflow-specific')}")
    if workflow=="Grounded RAG Chat":
        if r.get("llm_enabled"):
            st.success("Conversational AI: LLM reasoning enabled · evidence verification remains the final gate.")
        else:
            st.info("Conversational AI: deterministic fallback active. Set LLM_PROVIDER=openai and OPENAI_API_KEY to enable LLM reasoning.")
    if r.get("answer"): st.subheader("Answer"); st.write(r["answer"])
    if r.get("key_facts"):
        st.subheader("Key Facts"); st.dataframe(r["key_facts"],use_container_width=True)
    if r.get("authorities"):
        st.subheader("Research Authorities")
        for a in r["authorities"]:
            st.write({k:a.get(k) for k in ("case_name","court","date_year","citation","source_type","citation_status","source","relevant_passage")})
    if r.get("draft"):
        st.subheader("Grounded Working Draft"); st.code(r["draft"],language="text")
    if r.get("confidence") is not None: st.metric("Confidence",str(r["confidence"]).upper())
    claims(r.get("claims") or r.get("claim_verification")); sources(r.get("evidence") or r.get("sources") or r.get("authorities"))
    for title,key in [("Citations","citations"),("Contradictions","contradictions"),("Missing Information","missing_information"),("Warnings","warnings")]:
        vals=r.get(key) or []; st.subheader(title)
        if vals:
            for v in vals: st.write(v if isinstance(v,str) else v)
        else: st.info("None reported.")
    st.caption("Evidence-grounded prototype; not legal advice.")

if __name__=="__main__": main()
