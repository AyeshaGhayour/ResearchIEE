
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# Load a lightweight pretrained AI model
_model = None


def get_model():
    global _model

    if _model is None:
        print("Loading ResearchMind AI model...")
        _model = SentenceTransformer("all-MiniLM-L6-v2")

    return _model


def split_into_chunks(text, chunk_size=180, overlap=40):
    """
    Divide research text into smaller overlapping passages.
    """

    if not isinstance(text, str) or not text.strip():
        return []

    if chunk_size <= 0:
        raise ValueError("Chunk size must be greater than zero.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("Overlap must be non-negative and smaller than chunk size.")

    words = text.split()
    chunks = []

    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])

        if chunk.strip():
            chunks.append(chunk)

        if end >= len(words):
            break

        start = end - overlap

    return chunks

def prepare_paper_passages(paper_data, paper_name):
    """
    Convert extracted research sections into searchable passages.
    Preserve section headings, categories, and original page numbers.
    """

    from src.text_processor import clean_text

    passages = []

    sections = paper_data.get("sections", [])

    for section_data in sections:

        section_text = clean_text(
            section_data.get("text", "")
        )

        if not section_text:
            continue

        chunks = split_into_chunks(section_text)

        for chunk_number, chunk in enumerate(chunks, start=1):

            passages.append({
                "paper": paper_name,
                "page": section_data.get("page"),
                "heading": section_data.get("heading", ""),
                "section": section_data.get("section", "other"),
                "chunk": chunk_number,
                "text": chunk
            })

    return passages

def build_search_index(passages):
    """
    Generate AI embeddings for research passages.
    Keep embeddings aligned with the original passage list.
    """

    if not passages:
        return None

    texts = []

    for index, item in enumerate(passages):

        if not isinstance(item, dict):
            raise ValueError(
                f"Invalid passage at position {index + 1}."
            )

        text = item.get("text")

        if not isinstance(text, str) or not text.strip():
            raise ValueError(
                f"Missing or invalid text in passage {index + 1}."
            )

        texts.append(text)

    model = get_model()

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    return embeddings

    """
    Generate AI embeddings for valid research passages.
    """

    if not passages:
        return None

    valid_passages = [
        item for item in passages
        if isinstance(item.get("text"), str)
        and item["text"].strip()
    ]

    if not valid_passages:
        return None

    model = get_model()

    texts = [
        item["text"] for item in valid_passages
    ]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    return embeddings

    """
    Generate AI embeddings for research passages.
    """

    if not passages:
        return None

    model = get_model()

    texts = [item["text"] for item in passages]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    return embeddings


def search_papers(question, passages, embeddings, top_k=5):
    """
    Find research passages semantically related to a question.
    """

    if not isinstance(question, str) or not question.strip():
        return []

    if not passages or embeddings is None:
        return []

    if top_k <= 0:
        return []

    if len(passages) != embeddings.shape[0]:
        raise ValueError(
            "The number of passages does not match the search index."
        )

    model = get_model()

    question_embedding = model.encode(
        [question.strip()],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    similarities = cosine_similarity(
        question_embedding,
        embeddings
    )[0]

    ranked_indices = similarities.argsort()[::-1][:top_k]

    results = []

    for index in ranked_indices:
        result = passages[index].copy()

        result["score"] = round(
            float(similarities[index]) * 100, 2
        )

        results.append(result)

    return results
