from typing import List, Dict
import re

from src.text_cleaner import clean_text, detect_financial_section


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> List[str]:
    """Split into overlapping chunks, preferring sentence/word boundaries.

    Chunk size is approximate. A long unbroken token may exceed it rather than
    being cut in the middle. The final chunk always includes the end of the page.
    """
    if not text or not text.strip():
        return []
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Require chunk_size > 0 and 0 <= overlap < chunk_size")

    text = text.strip()
    chunks: List[str] = []
    start = 0
    length = len(text)

    while start < length:
        target = min(start + chunk_size, length)
        end = target
        if target < length:
            # Prefer sentence endings reasonably close to the target.
            region = text[start:target]
            sentence_ends = [m.end() for m in re.finditer(r"[.!?](?:\s+|$)", region)]
            candidates = [start + pos for pos in sentence_ends if pos >= int(len(region) * 0.60)]
            if candidates:
                end = candidates[-1]
            else:
                # Otherwise end at a whitespace boundary (never mid-word).
                boundary = text.rfind(" ", start + int(chunk_size * 0.60), target + 1)
                if boundary > start:
                    end = boundary + 1
                else:
                    # Very long word: include the complete token.
                    next_space = text.find(" ", target)
                    end = length if next_space < 0 else next_space + 1

        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= length:
            break

        next_start = max(start + 1, end - overlap)
        # Move the overlap start back to a word boundary.
        if next_start > 0 and not text[next_start - 1].isspace():
            boundary = text.rfind(" ", start + 1, next_start + 1)
            if boundary > start:
                next_start = boundary + 1
        if next_start >= end:
            next_start = end
        start = next_start

    return chunks


def create_chunks(pages: List[Dict], chunk_size: int = 900, overlap: int = 150) -> List[Dict]:
    """Create source/page/section-labelled chunks for retrieval."""
    all_chunks = []
    for page in pages:
        cleaned_text = clean_text(page["text"])
        page_chunks = chunk_text(cleaned_text, chunk_size=chunk_size, overlap=overlap)
        for chunk_index, chunk in enumerate(page_chunks):
            all_chunks.append({
                "chunk_id": f"{page['source']}_p{page['page']}_c{chunk_index}",
                "source": page["source"],
                "page": page["page"],
                "section": detect_financial_section(chunk),
                "text": chunk,
            })
    return all_chunks
