import re


def clean_text(text: str) -> str:
    """
    Cleans messy PDF text.

    PDF extraction often creates:
    - extra spaces
    - broken lines
    - repeated page labels
    - table of contents noise
    """

    if not text:
        return ""

    text = text.replace("\x00", " ")

    text = re.sub(r"\s+", " ", text)

    text = re.sub(
        r"Page \d+ of \d+",
        " ",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"Table of Contents",
        " ",
        text,
        flags=re.IGNORECASE
    )

    return text.strip()


def detect_financial_section(text: str) -> str:
    """
    Detects the rough financial section of a chunk.

    This makes the project finance-specific instead of generic RAG.
    """

    lower_text = text.lower()

    section_rules = {
        "risk_factors": [
            "risk factors",
            "risks related",
            "uncertainties",
            "could adversely affect",
            "material adverse"
        ],
        "management_discussion": [
            "management discussion",
            "md&a",
            "results of operations",
            "liquidity and capital resources"
        ],
        "financial_statements": [
            "consolidated statements",
            "balance sheet",
            "cash flows",
            "statement of operations",
            "income statement"
        ],
        "business_overview": [
            "business overview",
            "our business",
            "company overview",
            "we operate",
            "our products"
        ],
        "market_outlook": [
            "outlook",
            "guidance",
            "future expectations",
            "macroeconomic",
            "market conditions"
        ]
    }

    for section_name, keywords in section_rules.items():
        for keyword in keywords:
            if keyword in lower_text:
                return section_name

    return "general"