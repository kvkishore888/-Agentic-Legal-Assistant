"""Stateless HTTP prototype: PDFs stay within the current request."""
import base64
import json
import tempfile
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from app.backend_adapter import run_workflow
from ingestion.pdf_loader import load_documents
from ingestion.chunker import chunk_pages
from retrieval.keyword_search import KeywordIndex

class RequestRetriever:
    def __init__(self, chunks):
        self.index = KeywordIndex()
        self.index.add(chunks)
    def retrieve(self, query, top_k=5):
        return self.index.search(query, top_k)

class handler(BaseHTTPRequestHandler):
    def respond(self, code, data):
        payload = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(payload)
    def do_GET(self):
        self.respond(200, {'status':'ok','retrieval':'request-local BM25','llm':'optional'})
    def do_POST(self):
        try:
            size = int(self.headers.get('Content-Length', '0'))
            if not 0 < size <= 4_000_000:
                self.respond(413, {'error':'Upload request must be below 4 MB.'})
                return
            body = json.loads(self.rfile.read(size))
            files = body.get('files', [])
            if not isinstance(files, list) or not 1 <= len(files) <= 5:
                raise ValueError('Upload between 1 and 5 PDFs.')
            with tempfile.TemporaryDirectory() as tmp:
                paths = []
                for item in files:
                    name = item['name']
                    if Path(name).name != name or not name.lower().endswith('.pdf') or name in [Path(p).name for p in paths]:
                        raise ValueError('Use unique plain PDF filenames.')
                    data = base64.b64decode(item['data'], validate=True)
                    if not data.startswith(b'%PDF-'):
                        raise ValueError('File is not a PDF.')
                    path = Path(tmp) / name
                    path.write_bytes(data)
                    paths.append(str(path))
                chunks = chunk_pages(load_documents(paths, use_ocr=False))
            if not chunks:
                raise ValueError('No readable text. This web prototype requires text PDFs; use Streamlit with OCR for scans.')
            result = run_workflow(body.get('workflow','Grounded RAG Chat'), body.get('query',''),
                                  document_type=body.get('document_type','legal notice'),
                                  retriever=RequestRetriever(chunks))
            result['indexed_chunks'] = len(chunks)
            self.respond(200, result)
        except (ValueError, TypeError, KeyError):
            self.respond(400, {'error':'Invalid request. Check PDF files, question and workflow.'})
        except Exception:
            self.respond(500, {'error':'Could not process this document. Try another text PDF.'})
