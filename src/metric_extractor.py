import re
from typing import List, Dict


FINANCIAL_PATTERNS = {
    "money_values": r"\$\s?\d+(?:\.\d+)?\s?(?:million|billion|thousand|m|bn)?",
    "percentages": r"\d+(?:\.\d+)?\s?%",
    "years": r"\b20\d{2}\b",
    "quarters": r"\bQ[1-4]\b|first quarter|second quarter|third quarter|fourth quarter",
}


def extract_financial_metrics(chunks: List[Dict]) -> Dict:
    """
    Extracts simple financial mentions from retrieved chunks.

    Example:
    "$10 billion", "15%", "2023", "Q4"
    """

    combined_text = " ".join([chunk["text"] for chunk in chunks])

    results = {}

    for metric_name, pattern in FINANCIAL_PATTERNS.items():
        matches = re.findall(
            pattern,
            combined_text,
            flags=re.IGNORECASE
        )

        results[metric_name] = sorted(set(matches))[:20]

    return results