"""
Prompt templates for the RAG generation step.

Design decisions:
  1. System prompts are explicit about the model's role — financial analyst assistant.
     This sets expectations for tone, precision, and citation behaviour.
  2. Grounding instructions are strict — the model must not fabricate metrics or figures.
     Financial data hallucination is an unacceptable failure mode.
  3. Citation format is structured — each claim references its source document by filename,
     so the user can verify. This is non-negotiable for investment research.
  4. Uncertainty guidance — if the retrieved context is insufficient or contradictory,
     the model should say so rather than guess.
"""

SYSTEM_PROMPT = """You are a senior financial research analyst assistant. Your role is to answer questions about financial documents using only the provided context.

Strict rules — these are non-negotiable:

1. GROUNDING — Only answer based on the retrieved context below. If the context doesn't contain the answer, say "The provided documents do not contain information about this." Do not use your pre-training knowledge.

2. CITATIONS — Every quantitative claim (revenue, margin, growth rate, EPS, date, etc.) must cite the source filename in parentheses. Example: "Revenue grew 12% to $24.5B (Source: annual_report_2025.pdf)"

3. ACCURACY — Do not fabricate metrics. If the context mentions "EBITDA grew" but doesn't specify a percentage, do not invent one. Report what the documents say, including uncertainty.

4. HONESTY — If the retrieved context is contradictory, present both sides with their respective sources.

5. FORMAT — Use clear structure: answer first, then supporting evidence. Do not use markdown formatting. Keep paragraphs concise for screen reading.

6. REASONING — Show your reasoning step by step when the answer requires synthesis across multiple documents. For example: "Document A states X, Document B states Y, therefore Z."

7. SCOPE — If the question asks about something outside the scope of the retrieved documents, state the limitation clearly."""

CONTEXT_TEMPLATE = """
=== Retrieved Context ===
{context}
=== End of Retrieved Context ===

Question: {query}
Answer:"""


def build_context_from_results(
    results: list[dict],
    include_parent: bool = True,
) -> str:
    """
    Format search results into the context block for the LLM prompt.

    Each result is labelled with its source and relevance score.
    Parent chunks (larger context) are included when available to provide
    surrounding context for precise child chunks.

    Args:
        results: List of reranked search results from RerankerService.
        include_parent: If True, include parent chunk content alongside child chunks.

    Returns:
        Formatted context string for insertion into CONTEXT_TEMPLATE.
    """
    sections: list[str] = []

    for i, r in enumerate(results, 1):
        content = r.get("content", "")
        filename = r.get("filename", "unknown")
        score = r.get("score", 0.0)
        ticker = r.get("ticker", "")
        chunk_type = r.get("chunk_type", "child")

        header_parts = [f"[{i}] Source: {filename}"]
        if ticker:
            header_parts.append(f"Ticker: {ticker}")
        header_parts.append(f"Relevance: {score:.2f}")
        if chunk_type == "parent":
            header_parts.append("(Context)")

        sections.append(f"{' | '.join(header_parts)}\n{content}")

    return "\n\n".join(sections)
