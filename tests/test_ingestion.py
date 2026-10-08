from ingestion.chunker import chunk_pages
from ingestion.metadata import make_chunk_id
def test_chunk_metadata_is_preserved():
    pages=[{"text":"The accused was arrested on 10 March.","document":"FIR.pdf","page":2,"metadata":{"source_path":"/tmp/FIR.pdf"}}]
    c=chunk_pages(pages,chunk_size=100,overlap=10)[0]
    assert c["document"]=="FIR.pdf" and c["page"]==2 and c["chunk_id"]=="FIR_p2_c1"
    assert c["metadata"]["source_path"]=="/tmp/FIR.pdf"
def test_stable_chunk_id(): assert make_chunk_id("FIR.pdf",2,4)=="FIR_p2_c4"
