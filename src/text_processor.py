
import re
import unicodedata


def clean_text(text):
    """
    Clean extracted research text while preserving scientific
    symbols, Unicode characters, and meaningful punctuation.
    """

    if not text:
        return ""

    # Normalize Unicode characters
    text = unicodedata.normalize("NFKC", text)

    # Remove soft hyphens
    text = text.replace("\u00ad", "")

    # Remove unwanted control characters while preserving
    # normal whitespace
    text = "".join(
        char for char in text
        if unicodedata.category(char) != "Cc"
        or char in "\n\t\r"
    )

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def process_pages(pages):
    """
    Clean the text of every extracted PDF page.
    Preserve original page numbers.
    """

    processed_pages = []

    for page in pages:

        cleaned = clean_text(page.get("text", ""))

        if cleaned:
            processed_pages.append({
                "page": page["page"],
                "text": cleaned
            })

    return processed_pages
