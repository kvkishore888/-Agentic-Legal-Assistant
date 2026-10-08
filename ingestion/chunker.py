"""Legal-aware, page-preserving chunking."""
from .metadata import make_chunk_id, build_metadata

def chunk_pages(pages, chunk_size: int=1200, overlap: int=150):
    if chunk_size<=0 or overlap<0 or overlap>=chunk_size: raise ValueError("Invalid chunk parameters")
    out=[]
    for page in pages:
        text=(page.get("text") or "").strip()
        if not text: continue
        doc=page["document"]; num=int(page["page"]); section=page.get("section","")
        start=0; idx=1
        while start<len(text):
            end=min(len(text),start+chunk_size)
            if end<len(text):
                boundary=max(text.rfind("\n",start,end),text.rfind(". ",start,end))
                if boundary>start+chunk_size//2: end=boundary+1
            chunk=text[start:end].strip()
            if chunk:
                out.append({"text":chunk,"document":doc,"page":num,"section":section,
                            "chunk_id":make_chunk_id(doc,num,idx),
                            "metadata":build_metadata(doc,num,section)})
                idx+=1
            if end>=len(text): break
            start=max(0,end-overlap)
    return out
