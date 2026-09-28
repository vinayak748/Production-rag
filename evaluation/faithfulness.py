"""
LLM-as-judge faithfulness check.

Retrieval metrics (precision/recall/NDCG) tell you whether you retrieved
the right chunks. They don't tell you whether the final answer actually
USED that context, or whether the model quietly answered from its own
parametric knowledge instead (which looks fine until the context and the
model's prior disagree, and then it's wrong in a way retrieval metrics
can't catch).

This module scores an (answer, context) pair for groundedness using a
separate LLM call configured as a strict judge.
"""

import os
import json
from anthropic import Anthropic

JUDGE_PROMPT = """You are a strict evaluator. Given a QUESTION, the CONTEXT
that was provided to an AI assistant, and the assistant's ANSWER, judge
whether the answer is fully grounded in the context.

Respond with ONLY a JSON object, no other text:
{{"grounded": true/false, "reason": "one sentence explanation"}}

QUESTION: {question}

CONTEXT:
{context}

ANSWER: {answer}"""


def check_faithfulness(question: str, context: str, answer: str, model: str = "claude-sonnet-4-6") -> dict:
    client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    prompt = JUDGE_PROMPT.format(question=question, context=context, answer=answer)

    response = client.messages.create(
        model=model,
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response.content[0].text.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"grounded": None, "reason": f"Judge returned unparseable output: {raw}"}
