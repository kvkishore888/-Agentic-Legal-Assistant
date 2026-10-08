# Gemini on Render

Set GEMINI_API_KEY in the Render service Environment, then set LLM_PROVIDER=gemini.
GEMINI_MODEL defaults to gemini-3.8-flash; override it with a model available to your Google AI Studio project.
Save and redeploy after merging this change. The build installs google-genai.

If LLM_PROVIDER is absent or auto, a Gemini key selects Gemini before OpenAI.
Explicit none disables LLM calls; explicit openai keeps OpenAI selected.
GEMINI_MODEL is separate from LLM_MODEL so an existing OpenAI model setting does not break Gemini.

Only Grounded RAG Chat calls the LLM. Review, drafting and research remain deterministic workflows.
Retrieved evidence and recent chat history are sent to the selected provider when enabled.
All candidate claims still pass the existing grounding and citation gates. Provider failures,
empty or blocked responses use the deterministic fallback. A configured provider is not proof of
successful API access; inspect the response status banner. Do not commit API keys.
