import os
import ollama

SYSTEM_PROMPT = """You answer questions using ONLY the provided context.
If the context doesn't contain the answer, say so explicitly instead of
guessing. Cite which excerpt(s) you used."""

DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "llama3")


def generate_answer(query: str, context_chunks: list[str], model: str = DEFAULT_MODEL) -> str:
    context_block = "\n\n".join(
        f"[Excerpt {i+1}]\n{chunk}" for i, chunk in enumerate(context_chunks)
    )

    user_message = f"""Context:
{context_block}

Question: {query}"""

    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    )
    return response["message"]["content"]