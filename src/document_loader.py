from pathlib import Path
from typing import List, Dict
from pypdf import PdfReader


def load_pdf(file_path: str) -> List[Dict]:
    """
    Reads a PDF file page by page.
    Each page is stored with source name, page number, and text.
    """

    path = Path(file_path)
    reader = PdfReader(str(path))

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        pages.append({
            "source": path.name,
            "page": page_number,
            "text": text
        })

    return pages


def load_txt(file_path: str) -> List[Dict]:
    """
    Reads a TXT file.
    We treat the whole TXT file as page 1.
    """

    path = Path(file_path)

    text = path.read_text(
        encoding="utf-8",
        errors="ignore"
    )

    return [{
        "source": path.name,
        "page": 1,
        "text": text
    }]


def load_document(file_path: str) -> List[Dict]:
    """
    Chooses the correct loader based on file type.
    Supports PDF and TXT.
    """

    suffix = Path(file_path).suffix.lower()

    if suffix == ".pdf":
        return load_pdf(file_path)

    if suffix == ".txt":
        return load_txt(file_path)

    raise ValueError(f"Unsupported file type: {suffix}")