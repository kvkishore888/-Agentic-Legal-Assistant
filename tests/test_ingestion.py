from ingestion.chunker import chunk_pages
from ingestion.metadata import make_chunk_id

def test_chunk_metadata_is_preserved():
    pages=[{"text":"The accused was arrested on 10 March.","document":"FIR.pdf","page":2}]
    chunks=chunk_pages(pages,chunk_size=100,overlap=10)
    assert chunks[0]["document"]=="FIR.pdf"
    assert chunks[0]["page"]==2
    assert chunks[0]["chunk_id"]=="FIR_p2_c1"

def test_stable_chunk_id():
    assert make_chunk_id("FIR.pdf",2,4)=="FIR_p2_c4"
