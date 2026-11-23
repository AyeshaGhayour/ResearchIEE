
from app import all_passages, search_embeddings

from src.search_engine import search_papers
from src.research_analyzer import (
    analyze_connections,
    compare_papers,
    explore_research_gaps,
    generate_research_ideas
)

print("\n" + "=" * 60)
print("RESEARCHMIND FEATURE TESTS")
print("=" * 60)

# 1. Semantic Research Explorer
print("\n1. SEMANTIC RESEARCH EXPLORER")

results = search_papers(
    "How is machine learning used in healthcare?",
    all_passages,
    search_embeddings,
    top_k=3
)

print("Search results:", len(results))

# 2. Research Connection Map
print("\n2. RESEARCH CONNECTION MAP")

connections = analyze_connections(
    all_passages,
    top_n=5
)

print("Connections:", len(connections))

for item in connections[:3]:
    print(
        item["paper_1"],
        "<->",
        item["paper_2"],
        "| Score:",
        item["score"],
        "| Concept:",
        item["keyword"]
    )

# 3. Paper Comparison
print("\n3. PAPER COMPARISON")

paper_names = list(dict.fromkeys(
    passage["paper"] for passage in all_passages
))

if len(paper_names) >= 2:
    comparison = compare_papers(
        all_passages,
        paper_names[:2]
    )

    for name, data in comparison.items():
        print("\nPaper:", name)
        print("Available:", data["available"])
        print("Passages:", data["passage_count"])
        print("Objectives evidence:", len(data["objectives"]))
        print("Methodology evidence:", len(data["methodology"]))
        print("Findings evidence:", len(data["findings"]))
        print("Limitations evidence:", len(data["limitations"]))
        print("Future work evidence:", len(data["future_work"]))

# 4. Research Gap Explorer
print("\n4. RESEARCH GAP EXPLORER")

gaps = explore_research_gaps(
    all_passages,
    top_n=5
)

print("Potential gaps:", len(gaps))

for gap in gaps[:3]:
    print(
        gap["topic"],
        "| Coverage:",
        f'{gap["paper_count"]}/{gap["total_papers"]}'
    )

# 5. Idea Catalyst
print("\n5. IDEA CATALYST")

ideas = generate_research_ideas(
    gaps,
    all_passages,
    top_n=3
)

print("Ideas generated:", len(ideas))

for idea in ideas:
    print("\nTopic:", idea["topic"])
    print("Idea:", idea["idea"])
    print("Evidence:", len(idea["evidence"]))

print("\n" + "=" * 60)
print("ALL FEATURE TESTS COMPLETED")
print("=" * 60)
