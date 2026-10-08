"""Streamlit frontend for the integrated Agentic Legal Assistant."""
from __future__ import annotations
import sys
import tempfile
from pathlib import Path

# Streamlit executes this file from the app/ directory. Ensure the repository
# root is importable so the shared ingestion/retrieval/legal packages resolve.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
from ingestion.chunker import chunk_pages
from ingestion.pdf_loader import load_documents
from retrieval.hybrid_search import HybridRetriever,configure
from app.backend_adapter import run_workflow
st.set_page_config(page_title="Agentic Legal Assistant",page_icon="⚖️",layout="wide")
st.markdown("""<style>
.block-container{padding-top:1.5rem}.hero{padding:1.2rem;border:1px solid rgba(128,128,128,.25);border-radius:16px;margin-bottom:1rem}
.source{padding:.7rem;border:1px solid rgba(128,128,128,.2);border-radius:10px;margin:.4rem 0}
.status{font-weight:700}
</style>""",unsafe_allow_html=True)
WORKFLOWS=["Grounded RAG Chat","Case / Contract Review","Legal Drafting","Legal Research"]
def init():
    for k,v in {"documents":[],"chunks":[],"last_result":None}.items(): st.session_state.setdefault(k,v)
def process(files):
    with tempfile.TemporaryDirectory() as tmp:
        paths=[]
        for f in files:
            p=Path(tmp)/f.name;p.write_bytes(f.getvalue());paths.append(str(p))
        pages=load_documents(paths);chunks=chunk_pages(pages);r=HybridRetriever();r.index(chunks);configure(r)
        st.session_state.documents=[{"name":Path(p).name,"pages":sum(x["document"]==Path(p).name for x in pages)} for p in paths]
        st.session_state.chunks=chunks
def sources(items):
    st.subheader("Evidence & Sources")
    for x in items or []:
        if not isinstance(x,dict): continue
        st.markdown(f'<div class="source"><b>{x.get("document","Unknown")}</b> · Page {x.get("page","—")} · <code>{x.get("chunk_id","—")}</code><br>{x.get("text","")}</div>',unsafe_allow_html=True)
def claims(items):
    st.subheader("Claim Verification")
    if not items: st.info("No claim-level verification was returned.");return
    for x in items:
        if isinstance(x,str): st.write(x);continue
        st.write(f'**{x.get("status","UNKNOWN")}** — {x.get("claim_text") or x.get("claim") or x.get("text","")}')
def main():
    init()
    with st.sidebar:
        st.markdown("## ⚖️ Agentic Legal Assistant")
        files=st.file_uploader("Upload PDF legal documents",type=["pdf"],accept_multiple_files=True)
        if st.button("Process documents",type="primary",disabled=not files,use_container_width=True):
            try: process(files);st.success(f"Indexed {len(st.session_state.chunks)} chunks.")
            except Exception as e: st.error(f"Processing failed: {e}")
        workflow=st.selectbox("Workflow",WORKFLOWS)
        top_k=st.slider("Evidence sources",3,10,5)
        st.divider();st.write(f"Documents: **{len(st.session_state.documents)}**");st.write(f"Chunks: **{len(st.session_state.chunks)}**")
        for d in st.session_state.documents: st.caption(f"📄 {d['name']} · {d['pages']} pages")
    st.markdown('<div class="hero"><h1>Legal analysis you can verify.</h1><p>Evidence first: retrieve → reason → verify → cite. Unsupported claims stay visible.</p></div>',unsafe_allow_html=True)
    prompts={"Grounded RAG Chat":"Ask a question about the uploaded documents","Case / Contract Review":"Describe the case or contract issue","Legal Drafting":"Describe the document you need drafted","Legal Research":"Ask a legal research question"}
    q=st.text_area(prompts[workflow],height=110)
    if st.button("Run analysis",type="primary",use_container_width=True,disabled=not q.strip()):
        if not st.session_state.chunks: st.warning("Process at least one document first.");return
        try:
            with st.spinner("Running integrated workflow…"): st.session_state.last_result=run_workflow(workflow,q,top_k)
        except Exception as e: st.error(f"Analysis failed: {e}");return
    r=st.session_state.last_result
    if not r: st.info("Upload documents, choose a workflow, and run an analysis.");return
    st.divider();st.caption(f"Route: {r.get('route','workflow-specific')}")
    if r.get("answer"): st.subheader("Answer");st.write(r["answer"])
    if r.get("draft"): st.subheader("Draft");st.code(r["draft"],language="text")
    if r.get("confidence") is not None: st.metric("Confidence",str(r["confidence"]).upper())
    claims(r.get("claims") or r.get("claim_verification"))
    sources(r.get("evidence") or r.get("sources") or r.get("authorities"))
    for title,key in [("Citations","citations"),("Contradictions","contradictions"),("Missing Information","missing_information"),("Warnings","warnings")]:
        vals=r.get(key) or []
        st.subheader(title)
        if vals:
            for v in vals: st.write(v if isinstance(v,str) else v)
        else: st.info("None reported.")
    st.caption("Prototype for grounded legal assistance; not legal advice.")
if __name__=="__main__": main()
