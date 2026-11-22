import sys
import time
import os
import requests
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from src.rag_pipeline import FinSightRAG


TEST_QUESTIONS = [
    {
        "question": "Compare the main risk factors across Apple, Microsoft, and NVIDIA.",
        "expected_sources": ["apple_10k_2024.pdf", "microsoft_10k_2024.pdf", "nvidia_10k_2024.pdf"],
        "evaluation_criteria": (
            "A good answer should identify distinct risk factors for each company "
            "(e.g. supply chain for Apple, regulatory/AI risks for Microsoft, "
            "demand concentration for NVIDIA) and cite sources."
        )
    },
    {
        "question": "Compare the main revenue drivers for Apple, Microsoft, and NVIDIA.",
        "expected_sources": ["apple_10k_2024.pdf", "microsoft_10k_2024.pdf", "nvidia_10k_2024.pdf"],
        "evaluation_criteria": (
            "A good answer should name the specific revenue segments for each company "
            "(e.g. iPhone/Services for Apple, Azure/Cloud for Microsoft, "
            "Data Center for NVIDIA) with supporting evidence."
        )
    },
    {
        "question": "Compare the cloud, AI, and data center strategy across Microsoft and NVIDIA.",
        "expected_sources": ["microsoft_10k_2024.pdf", "nvidia_10k_2024.pdf"],
        "evaluation_criteria": (
            "A good answer should cover Microsoft Azure and Copilot/OpenAI partnership, "
            "and NVIDIA's GPU/accelerated computing focus. "
            "It should distinguish the two companies' different roles in AI infrastructure."
        )
    },
    {
        "question": "What financial metrics are mentioned across the uploaded filings?",
        "expected_sources": [],
        "evaluation_criteria": (
            "A good answer should mention specific numbers such as revenue figures, "
            "gross margins, or net income, with at least one concrete figure per company."
        )
    }
]


# ---------------------------------------------------------------------------
# LLM Judge
# ---------------------------------------------------------------------------

def score_answer_with_llm(question: str, answer: str, criteria: str) -> dict:
    """
    Uses your configured LLM (Gemini or Ollama) to score the answer quality.

    Returns a dict with:
        - score (1-5)
        - reasoning (short explanation)

    This is called an 'LLM-as-judge' pattern. The same LLM that generates
    answers is asked to evaluate them against defined criteria. It is not
    perfect, but it is far more meaningful than measuring answer length alone.
    """

    judge_prompt = f"""You are an expert financial analyst evaluating an AI-generated answer.

Question asked:
{question}

Answer to evaluate:
{answer}

Evaluation criteria:
{criteria}

Score the answer on a scale of 1 to 5:
1 = Very poor (missing key information, vague, or inaccurate)
2 = Poor (some relevant content but major gaps)
3 = Acceptable (covers basics but lacks depth or specifics)
4 = Good (solid answer with relevant details and sources cited)
5 = Excellent (comprehensive, specific, well-sourced, clearly organized)

Respond in this exact format:
SCORE: <number 1-5>
REASONING: <one or two sentences explaining the score>
"""

    provider = os.getenv("LLM_PROVIDER", "ollama").lower()
    gemini_key = os.getenv("GEMINI_API_KEY")
    gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    ollama_model = os.getenv("OLLAMA_MODEL", "llama3.1:8b")

    raw_response = None

    if provider == "gemini" and gemini_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=gemini_key)
            model = genai.GenerativeModel(gemini_model)
            response = model.generate_content(judge_prompt)
            raw_response = response.text
        except Exception as e:
            raw_response = None

    if raw_response is None:
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": ollama_model,
                    "prompt": judge_prompt,
                    "stream": False,
                    "options": {"temperature": 0.0}
                },
                timeout=120
            )
            response.raise_for_status()
            raw_response = response.json()["response"]
        except Exception as e:
            return {"score": "N/A", "reasoning": f"LLM judge unavailable: {e}"}

    # Parse the response
    score = "N/A"
    reasoning = "Could not parse LLM response."

    if raw_response:
        for line in raw_response.splitlines():
            line = line.strip()
            if line.upper().startswith("SCORE:"):
                try:
                    score = int(line.split(":", 1)[1].strip()[0])
                except (ValueError, IndexError):
                    pass
            elif line.upper().startswith("REASONING:"):
                reasoning = line.split(":", 1)[1].strip()

    return {"score": score, "reasoning": reasoning}


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate():
    rag = FinSightRAG()
    lines = []

    output_path = PROJECT_ROOT / "evaluation" / "evaluation_results.txt"

    def write(text=""):
        lines.append(text)
        print(text)

    write("FinSight RAG Evaluation Results")
    write("=" * 80)
    write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    write()

    write("Loading saved knowledge base...")
    stats = rag.load_existing_index()
    write(str(stats))
    write()

    results = []

    for test in TEST_QUESTIONS:
        question = test["question"]
        expected_sources = test["expected_sources"]
        criteria = test["evaluation_criteria"]

        write(f"Running: {question}")

        start_time = time.time()
        result = rag.ask(question)
        latency = round(time.time() - start_time, 2)

        retrieved_sources = sorted(
            set(source["source"] for source in result["sources"])
        )

        if expected_sources:
            coverage = sum(
                source in retrieved_sources
                for source in expected_sources
            ) / len(expected_sources)
        else:
            coverage = 1.0 if retrieved_sources else 0.0

        write("  Scoring answer quality...")
        quality = score_answer_with_llm(
            question=question,
            answer=result["answer"],
            criteria=criteria
        )

        row = {
            "question": question,
            "latency_seconds": latency,
            "retrieved_sources": retrieved_sources,
            "source_coverage": round(coverage, 2),
            "answer_length": len(result["answer"]),
            "num_chunks": len(result["sources"]),
            "quality_score": quality["score"],
            "quality_reasoning": quality["reasoning"]
        }

        results.append(row)

    write()
    write("Evaluation Results")
    write("=" * 80)

    for row in results:
        write()
        write(f"Question    : {row['question']}")
        write(f"Latency     : {row['latency_seconds']} seconds")
        write(f"Sources     : {row['retrieved_sources']}")
        write(f"Coverage    : {row['source_coverage']}")
        write(f"Answer len  : {row['answer_length']} characters")
        write(f"Chunks used : {row['num_chunks']}")
        write(f"Quality     : {row['quality_score']} / 5")
        write(f"Reasoning   : {row['quality_reasoning']}")

    numeric_scores = [
        r["quality_score"] for r in results
        if isinstance(r["quality_score"], int)
    ]

    avg_coverage = sum(r["source_coverage"] for r in results) / len(results)
    avg_latency = sum(r["latency_seconds"] for r in results) / len(results)
    avg_quality = (
        round(sum(numeric_scores) / len(numeric_scores), 2)
        if numeric_scores else "N/A"
    )

    write()
    write("Summary")
    write("=" * 80)
    write(f"Average source coverage : {round(avg_coverage, 2)}")
    write(f"Average latency         : {round(avg_latency, 2)} seconds")
    write(f"Average quality score   : {avg_quality} / 5")
    write()
    write(f"Saved to: {output_path}")

    output_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    evaluate()
