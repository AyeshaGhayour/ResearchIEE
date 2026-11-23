
import os

from app import process_paper
from src.search_engine import (
    build_search_index,
    search_papers
)

UPLOAD_FOLDER = "data/uploads"

all_passages = []

# ==========================================
# 1. Process all uploaded PDFs
# ==========================================

for filename in sorted(os.listdir(UPLOAD_FOLDER)):

    if not filename.lower().endswith(".pdf"):
        continue

    file_path = os.path.join(UPLOAD_FOLDER, filename)

    print("\n" + "=" * 60)
    print("PROCESSING:", filename)
    print("=" * 60)

    try:
        paper_info, passages = process_paper(
            file_path,
            filename
        )

        all_passages.extend(passages)

        print("Paper information:", paper_info)
        print("Passages:", len(passages))

    except Exception as error:
        print("ERROR:", error)


# ==========================================
# 2. Build semantic search index
# ==========================================

print("\n" + "=" * 60)
print("BUILDING SEARCH INDEX")
print("=" * 60)

embeddings = build_search_index(all_passages)

if embeddings is None:
    raise ValueError("Search index could not be created.")

print("Total passages:", len(all_passages))
print("Embedding shape:", embeddings.shape)


# ==========================================
# 3. Test semantic search
# ==========================================

question = "How is machine learning used in healthcare?"

print("\n" + "=" * 60)
print("SEMANTIC SEARCH TEST")
print("=" * 60)

print("Question:", question)

results = search_papers(
    question,
    all_passages,
    embeddings,
    top_k=5
)


# ==========================================
# 4. Display search results and metadata
# ==========================================

for number, result in enumerate(results, start=1):

    print("\nRESULT:", number)
    print("-" * 50)

    print("Paper:", result.get("paper"))
    print("Page:", result.get("page"))
    print("Heading:", result.get("heading"))
    print("Section:", result.get("section"))
    print("Chunk:", result.get("chunk"))
    print("Similarity:", result.get("score"), "%")
    print("Text:", result.get("text", "")[:300])


# ==========================================
# 5. Verify metadata
# ==========================================

print("\n" + "=" * 60)
print("METADATA VERIFICATION")
print("=" * 60)

required_fields = [
    "paper",
    "page",
    "heading",
    "section",
    "chunk",
    "text",
    "score"
]

for number, result in enumerate(results, start=1):

    missing_fields = [
        field for field in required_fields
        if field not in result
    ]

    if missing_fields:
        print(
            f"Result {number}: MISSING {missing_fields}"
        )
    else:
        print(f"Result {number}: ALL METADATA PRESENT")

print("\nSEARCH TEST COMPLETED.")
