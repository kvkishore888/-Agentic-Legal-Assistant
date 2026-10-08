"""Persistent Chroma adapter preserving retrieval metadata."""
import json,os
from pathlib import Path
class VectorStore:
    def __init__(self,path=None,collection=None):
        try: import chromadb
        except ImportError as exc: raise RuntimeError("Install chromadb for vector storage") from exc
        self.client=chromadb.PersistentClient(path=str(Path(path or os.getenv("VECTOR_DB_PATH",".legal_chroma"))))
        self.collection=self.client.get_or_create_collection(name=collection or os.getenv("VECTOR_COLLECTION","legal_chunks"),metadata={"hnsw:space":"cosine"})
    @staticmethod
    def _clean(meta):
        out={}
        for k,v in (meta or {}).items():
            if v is None: continue
            out[k]=v if isinstance(v,(str,int,float,bool)) else json.dumps(v,ensure_ascii=False)
        return out
    def upsert(self,chunks,embeddings):
        if len(chunks)!=len(embeddings): raise ValueError("chunks and embeddings must have the same length")
        if not chunks:return
        self.collection.upsert(ids=[c["chunk_id"] for c in chunks],documents=[c["text"] for c in chunks],embeddings=embeddings,
          metadatas=[self._clean({**(c.get("metadata") or {}),"document":c["document"],"page":c["page"],"section":c.get("section",""),"chunk_id":c["chunk_id"]}) for c in chunks])
    def search(self,query_embedding,top_k=5,where=None):
        if top_k<=0:return []
        r=self.collection.query(query_embeddings=[query_embedding],n_results=top_k,where=where,include=["documents","metadatas","distances"])
        rows=[]
        for d,m,cid,dist in zip((r.get("documents") or [[]])[0],(r.get("metadatas") or [[]])[0],(r.get("ids") or [[]])[0],(r.get("distances") or [[]])[0]):
            meta=dict(m or {}); rows.append({"text":d,"document":meta.get("document",""),"page":meta.get("page"),"section":meta.get("section",""),"chunk_id":cid,"score":1-float(dist),"metadata":meta})
        return rows