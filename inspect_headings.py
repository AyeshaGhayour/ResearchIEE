
import os
import pymupdf
from collections import Counter

UPLOAD_FOLDER = "data/uploads"

for filename in os.listdir(UPLOAD_FOLDER):
    if not filename.lower().endswith(".pdf"):
        continue

    path = os.path.join(UPLOAD_FOLDER, filename)

    with pymupdf.open(path) as document:
        print("\n" + "=" * 65)
        print("PAPER:", filename)
        print("=" * 65)

        for page_number in [1, 2, 3, 4]:
            if page_number > len(document):
                continue

            page = document[page_number - 1]
            lines = []

            for block in page.get_text("dict")["blocks"]:
                if "lines" not in block:
                    continue

                for line in block["lines"]:
                    spans = line["spans"]
                    text = " ".join(
                        s["text"].strip()
                        for s in spans if s["text"].strip()
                    )

                    if not text:
                        continue

                    size = max(s["size"] for s in spans)

                    bold = any(
                        s["flags"] & 16
                        for s in spans if s["text"].strip()
                    )

                    lines.append((round(size, 1), bold, text))

            print(f"\nPAGE {page_number}")
            print("Font size distribution:")

            counts = Counter(size for size, _, _ in lines)
            for size, count in counts.most_common(10):
                print(f"Size {size}: {count} lines")

            print("\nFirst 15 extracted lines:")
            for size, bold, text in lines[:15]:
                print(
                    f"Size={size} | Bold={bold} | {text[:110]}"
                )
