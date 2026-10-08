"""Legal-aware chunking that preserves all source metadata."""
from .metadata import make_chunk_id
def _boundary(text,start,end):
    if end>=len(text): return end
    point=max(text.rfind("\n",start,end),text.rfind(". ",start,end),text.rfind("; ",start,end),text.rfind(": ",start,end))
    return point+1 if point>start+(end-start)//2 else end
def chunk_pages(pages,chunk_size:int=1200,overlap:int=180):
    if chunk_size<=0 or overlap<0 or overlap>=chunk_size: raise ValueError("Invalid chunk parameters")
    out=[]
    for page in pages:
        text=(page.get("text") or "").strip()
        if not text: continue
        doc,num=page["document"],int(page["page"]); base=dict(page.get("metadata") or {})
        base.update({"document":doc,"page":num,"section":page.get("section","") or ""})
        start,idx=0,1
        while start<len(text):
            end=_boundary(text,start,min(len(text),start+chunk_size)); chunk=text[start:end].strip()
            if chunk:
                cid=make_chunk_id(doc,num,idx); meta=dict(base); meta["chunk_id"]=cid
                out.append({"text":chunk,"document":doc,"page":num,"section":page.get("section","") or "","chunk_id":cid,"metadata":meta}); idx+=1
            if end>=len(text): break
            start=max(0,end-overlap)
    return out