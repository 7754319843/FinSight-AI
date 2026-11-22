import os
import re
import requests
from dotenv import load_dotenv

load_dotenv()


class LLMGenerator:
    """
    Generates final answers using:
    1. Gemini if available
    2. Ollama local model if available
    3. Evidence-only offline fallback
    """

    def __init__(self):
        self.provider = os.getenv("LLM_PROVIDER", "ollama").lower()
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        self.ollama_model = os.getenv("OLLAMA_MODEL", "gemma3:4b")

    def generate(self, question: str, context_chunks):
        prompt = self._build_prompt(question, context_chunks)

        if self.provider == "gemini":
            try:
                answer = self._generate_gemini(prompt)
                return self.clean_output(answer)
            except Exception:
                pass

        try:
            answer = self._generate_ollama(prompt)
            return self.clean_output(answer)
        except Exception:
            answer = self._generate_offline(question, context_chunks)
            return self.clean_output(answer)

    def _build_prompt(self, question: str, context_chunks):
        context_text = ""

        for i, chunk in enumerate(context_chunks, start=1):
            context_text += (
                f"\nSOURCE {i}\n"
                f"Document: {chunk['source']}\n"
                f"Page: {chunk['page']}\n"
                f"Section: {chunk.get('section', 'general')}\n"
                f"Text: {chunk['text']}\n"
            )

        prompt = f"""
You are FinSight RAG, a financial research assistant.

Use ONLY the retrieved evidence below.

IMPORTANT FORMATTING RULES:
- Use bullet points only.
- Do not write dense paragraphs.
- Do not use markdown italics.
- Do not use asterisks.
- Do not merge words together.
- Always put spaces between numbers, years, and units.
- Correct: 245.1 billion in 2024.
- Wrong: 245.1billionin2024.
- Keep each bullet short.
- Mention source numbers like (Source 1).
- If evidence is missing, say "Not enough evidence in the retrieved documents."
- Do not ask follow-up questions.

STRICT ANSWER FORMAT:

Company or Topic:
- Point 1 with source.
- Point 2 with source.
- Point 3 with source.

Company or Topic:
- Point 1 with source.
- Point 2 with source.

User question:
{question}

Retrieved evidence:
{context_text}

Final answer:
"""
        return prompt

    def _generate_gemini(self, prompt: str):
        import google.generativeai as genai

        if not self.gemini_api_key:
            raise ValueError("Gemini API key not found.")

        genai.configure(api_key=self.gemini_api_key)
        model = genai.GenerativeModel(self.gemini_model)
        response = model.generate_content(prompt)

        return response.text

    def _generate_ollama(self, prompt: str):
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": self.ollama_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "top_p": 0.8
                }
            },
            timeout=120
        )

        response.raise_for_status()
        return response.json()["response"]

    @staticmethod
    def _readable_excerpt(text: str, max_chars: int = 900) -> str:
        """Avoid truncating excerpts mid-word; prefer a complete sentence."""
        text = " ".join(str(text).split())
        if len(text) <= max_chars:
            return text
        prefix = text[:max_chars]
        sentence_matches = list(re.finditer(r"[.!?](?=\s|$)", prefix))
        if sentence_matches and sentence_matches[-1].end() >= max_chars * 0.55:
            return prefix[:sentence_matches[-1].end()].rstrip()
        last_space = prefix.rfind(" ")
        if last_space > 0:
            return prefix[:last_space].rstrip() + "…"
        return prefix.rstrip() + "…"

    def _generate_offline(self, question: str, context_chunks):
        answer = f"Question: {question}\n\n"
        answer += "Evidence-based fallback answer from retrieved financial document chunks:\n\n"

        for i, chunk in enumerate(context_chunks[:9], start=1):
            snippet = self._readable_excerpt(chunk["text"], max_chars=900)

            answer += (
                f"Source {i}: {chunk['source']}, page {chunk['page']}, "
                f"section: {chunk.get('section', 'general')}\n"
                f"{snippet}\n\n"
            )

        return answer

    def clean_output(self, text: str) -> str:
        """
        Light cleanup only. Main formatting is controlled by the prompt.
        """

        if not text:
            return ""

        text = text.replace("*", "")
        text = text.replace("_", " ")

        text = re.sub(r"Source\s*(\d+)", r"Source \1", text, flags=re.IGNORECASE)
        text = re.sub(r"\(\s*Source", r"(Source", text)

        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()