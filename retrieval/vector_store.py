"""Persistent Chroma vector store adapter."""
from pathlib import Path
class VectorStore:
    def __init__(self, path=".legal_chroma", collection="legal_chunks"):
        try: import chromadb
        except ImportError as exc: raise RuntimeError("Install chromadb for vector storage") from exc
        self.client=chromadb.PersistentClient(path=str(Path(path)))
        self.collection=self.client.get_or_create_collection(collection, metadata={"hnsw:space":"cosine"})
    def upsert(self, chunks, embeddings):
        self.collection.upsert(ids=[c["chunk_id"] for c in chunks],
          documents=[c["text"] for c in chunks], embeddings=embeddings,
          metadatas=[{"document":c["document"],"page":c["page"],"section":c.get("section","")} for c in chunks])
    def search(self, query_embedding, top_k=5, where=None):
        r=self.collection.query(query_embeddings=[query_embedding],n_results=top_k,where=where)
        docs=(r.get("documents") or [[]])[0]; metas=(r.get("metadatas") or [[]])[0]
        ids=(r.get("ids") or [[]])[0]; ds=(r.get("distances") or [[]])[0]
        return [{"text":d,"document":m.get("document",""),"page":m.get("page"),
                 "chunk_id":cid,"score":1-float(dist)} for d,m,cid,dist in zip(docs,metas,ids,ds)]
