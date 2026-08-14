import os
from groq import Groq
from typing import List, Dict, Any, Tuple


SYSTEM_PROMPT = """You are an ultra-strict Document QA Assistant. 
Your single priority is absolute precision grounded strictly in the provided context documents.

RULES:
1. Answer the user question relying ONLY on the provided context passages.
2. Do NOT use outside knowledge, assumptions, or external information.
3. Every claim in your answer MUST be accompanied by an inline citation using the exact context tag, e.g., [Page 4, Chunk 2].
4. CRITICAL FALLBACK: If the answer cannot be explicitly derived from the provided context, output EXACTLY this string and nothing else:
   "Information not found in the provided document."
"""

class RAGEngine:
    def __init__(self, api_key: str = None, model_name: str = "llama-3.3-70b-versatile"):
        """
        Initializes Groq client with API key and selects Llama 3.3 70B model.
        """
        self.client = Groq(api_key=api_key or os.getenv("GROQ_API_KEY"))
        self.model_name = model_name

    def generate_grounded_response(
        self, query: str, context_chunks: List[Dict[str, Any]]
    ) -> Tuple[str, List[Dict[str, Any]]]:
        if not context_chunks:
            return "Information not found in the provided document.", []

        formatted_context_blocks = []
        for i, chunk in enumerate(context_chunks, 1):
            meta = chunk["metadata"]
            citation_tag = f"<sup>[Page {meta['page']}, Chunk {meta['chunk_index']}]</sup>"
            block = f"--- CONTEXT BLOCK {i} {citation_tag} ---\n{chunk['text']}\n"
            formatted_context_blocks.append(block)

        full_context_str = "\n".join(formatted_context_blocks)

        user_prompt = f"""CONTEXT PASSAGES:
{full_context_str}

USER QUESTION:
{query}

Provide a grounded response with inline citations:"""

        response = self.client.chat.completions.create(
            model=self.model_name,
            temperature=0.0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ]
        )

        answer = response.choices[0].message.content
        return answer, context_chunks