
from collections import defaultdict
from app import uploaded_papers, all_passages

print("\n" + "=" * 65)
print("UNCLASSIFIED HEADINGS")
print("=" * 65)

for paper in uploaded_papers:

    paper_name = paper["name"]

    print(f"\nPAPER: {paper_name}")

    headings = defaultdict(set)

    for passage in all_passages:

        if passage.get("paper") != paper_name:
            continue

        if passage.get("section") == "other":

            heading = passage.get("heading", "").strip()

            if heading:
                headings[heading].add(passage.get("page"))

    for heading, pages in sorted(headings.items()):

        page_list = ", ".join(
            str(page) for page in sorted(pages)
        )

        print(f"Page {page_list} | {heading}")
