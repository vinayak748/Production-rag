"""LLM generation over retrieved context. Kept separate from retrieval so
you can evaluate retrieval quality independently of generation quality —
a common mistake is to only judge RAG by "does the final answer look right,"
which hides whether retrieval or generation was the actual problem."""

import os
from anthropic import Anthropic

SYSTEM_PROMPT = """You answer questions using ONLY the provided context.
If the context doesn't contain the answer, say so explicitly instead of
guessing. Cite which excerpt(s) you used."""


def generate_answer(query: str, context_chunks: list[str], model: str = "claude-sonnet-4-6") -> str:
    client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    context_block = "\n\n".join(
        f"[Excerpt {i+1}]\n{chunk}" for i, chunk in enumerate(context_chunks)
    )

    user_message = f"""Context:
{context_block}

Question: {query}"""

    response = client.messages.create(
        model=model,
        max_tokens=500,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )
    return response.content[0].text
