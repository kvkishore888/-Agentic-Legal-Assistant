import base64
import json
import threading
from http.server import HTTPServer
from urllib.request import Request, urlopen
import fitz
from api.index import handler
import pytest

@pytest.mark.parametrize('workflow', ['Grounded RAG Chat','Case / Contract Review','Legal Drafting','Legal Research'])
def test_uploaded_pdf_to_grounded_workflow(workflow):
    pdf = fitz.open()
    page = pdf.new_page()
    page.insert_text((72,72), 'The contract value is 240000 rupees. Payment is due on 20 February 2026.')
    data = base64.b64encode(pdf.tobytes()).decode()
    pdf.close()
    server = HTTPServer(('127.0.0.1',0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        payload = json.dumps({'files':[{'name':'contract.pdf','data':data}], 'query':'What is the contract value?', 'workflow':workflow}).encode()
        request = Request(f'http://127.0.0.1:{server.server_port}/api/analyze', data=payload, headers={'Content-Type':'application/json'})
        with urlopen(request) as response:
            result = json.load(response)
        assert result['indexed_chunks'] == 1
        assert result['evidence'][0]['document'] == 'contract.pdf'
        if workflow == 'Grounded RAG Chat':
            assert '240000' in result['answer']
            assert result['citations'][0]['valid']
        if workflow == 'Legal Drafting':
            assert '240000' in result['draft']
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
