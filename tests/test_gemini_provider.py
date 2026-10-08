from types import SimpleNamespace
import pytest
from agent.agent import answer_with_grounding, llm_configuration

EVIDENCE = [{'text':'The contract value is 240000 rupees.', 'document':'a.pdf','page':1,'chunk_id':'a'}]

@pytest.fixture(autouse=True)
def clear_provider_env(monkeypatch):
    for name in ('LLM_PROVIDER','GEMINI_API_KEY','OPENAI_API_KEY','GEMINI_MODEL','LLM_MODEL'):
        monkeypatch.delenv(name, raising=False)

class FakeClient:
    text = 'The contract value is 240000 rupees.'
    failure = False
    calls = []
    def __init__(self, **kwargs):
        assert kwargs['api_key'] == 'fake-test-key'
        self.models = self
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        if self.failure: raise RuntimeError('provider failure')
        return SimpleNamespace(text=self.text)

@pytest.fixture
def gemini(monkeypatch):
    from google import genai
    monkeypatch.setenv('GEMINI_API_KEY','fake-test-key')
    monkeypatch.setattr(genai,'Client',FakeClient)
    monkeypatch.setattr(FakeClient,'calls',[])
    monkeypatch.setattr(FakeClient,'failure',False)
    monkeypatch.setattr(FakeClient,'text','The contract value is 240000 rupees.')
    return FakeClient

def test_auto_selects_gemini_key_without_other_config(monkeypatch):
    monkeypatch.setenv('GEMINI_API_KEY','fake-test-key')
    assert llm_configuration() == ('gemini', True)
    monkeypatch.setenv('LLM_PROVIDER','none')
    assert llm_configuration() == ('none', False)

def test_explicit_openai_not_overridden(monkeypatch):
    monkeypatch.setenv('GEMINI_API_KEY','fake-test-key')
    monkeypatch.setenv('LLM_PROVIDER','openai')
    assert llm_configuration() == ('openai', False)

def test_gemini_generation_history_and_grounding(gemini,monkeypatch):
    monkeypatch.setenv('GEMINI_MODEL','gemini-custom')
    monkeypatch.setenv('LLM_MODEL','openai-custom')
    result = answer_with_grounding('contract value', EVIDENCE,
        conversation_history=[{'role':'user','content':'Earlier question'},{'role':'assistant','content':'Earlier answer'}])
    assert result['llm_provider'] == 'gemini' and result['llm_enabled'] and result['llm_used']
    assert result['confidence'] == 'HIGH' and result['citations'][0]['valid']
    request = gemini.calls[0]
    assert request['model'] == 'gemini-custom'
    assert [x.role for x in request['contents']] == ['user','model','user']
    assert '240000' in request['contents'][-1].parts[0].text
    assert request['config'].system_instruction

def test_gemini_cannot_bypass_grounding(gemini,monkeypatch):
    monkeypatch.setattr(gemini,'text','The contract value is 999999 rupees.')
    result = answer_with_grounding('contract value',EVIDENCE)
    assert result['llm_used'] and result['confidence'] == 'LOW'
    assert '999999' not in result['answer']

@pytest.mark.parametrize('failure,text',[(True,''),(False,''),(False,None)])
def test_gemini_failure_or_empty_response_falls_back(gemini,monkeypatch,failure,text):
    monkeypatch.setattr(gemini,'failure',failure)
    monkeypatch.setattr(gemini,'text',text)
    result = answer_with_grounding('contract value',EVIDENCE)
    assert result['llm_enabled'] and not result['llm_used']
    assert result['answer'] == 'The contract value is 240000 rupees.'

def test_missing_key_never_calls_provider():
    assert not answer_with_grounding('contract value',EVIDENCE)['llm_enabled']
