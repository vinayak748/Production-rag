"""LLM generation over retrieved context. Kept separate from retrieval so
you can evaluate retrieval quality independently of generation quality —
a common mistake is to only judge RAG by "does the final answer look right,"
which hides whether retrieval or generation was the actual problem.

Two interchangeable backends, selected by GEN_BACKEND env var:
  - "anthropic" (default): needs ANTHROPIC_API_KEY, paid credits
  - "gemini": needs GEMINI_API_KEY, free tier available
"""

import os

SYSTEM_PROMPT = """You answer questions using ONLY the provided context.
If the context doesn't contain the answer, say so explicitly instead of
guessing. Cite which excerpt(s) you used."""


def _build_prompt(query: str, context_chunks: list[str]) -> str:
    context_block = "\n\n".join(
        f"[Excerpt {i+1}]\n{chunk}" for i, chunk in enumerate(context_chunks)
    )
    return f"""Context:
{context_block}

Question: {query}"""


def _generate_anthropic(query: str, context_chunks: list[str], model: str) -> str:
    from anthropic import Anthropic

    client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    response = client.messages.create(
        model=model or "claude-sonnet-4-6",
        max_tokens=500,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": _build_prompt(query, context_chunks)}],
    )
    return response.content[0].text


def _generate_gemini(query: str, context_chunks: list[str], model: str) -> str:
    from google import genai

    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    response = client.models.generate_content(
        model=model or "gemini-3.8-flash",
        contents=_build_prompt(query, context_chunks),
        config={"system_instruction": SYSTEM_PROMPT, "max_output_tokens": 500},
    )
    return response.text


def generate_answer(query: str, context_chunks: list[str], model: str | None = None) -> str:
    backend = os.environ.get("GEN_BACKEND", "anthropic").strip().lower()
    if backend == "gemini":
        return _generate_gemini(query, context_chunks, model)
    return _generate_anthropic(query, context_chunks, model)
