
from collections import Counter, defaultdict
from itertools import combinations
import re

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity

from src.search_engine import get_model


# ============================================================
# 1. KEYWORD AND PHRASE EXTRACTION
# ============================================================

STOP_WORDS = set(ENGLISH_STOP_WORDS) | {
    "study", "studies", "research", "paper", "papers",
    "results", "proposed", "method", "methods", "approach",
    "figure", "table", "using", "used", "based", "also",
    "however", "therefore", "author", "authors", "copyright",
    "license", "published", "article", "available", "abstract",
    "introduction", "background", "conclusion", "references",
    "significant", "respectively", "et", "al", "one", "two",
    "three", "first", "second", "third", "new", "different",
    "many", "much", "within", "without", "current", "generated",
    "analysed", "analyzed", "declarations", "materials",
    "provided", "further", "various", "including", "example",
    "examples", "section", "sections", "following", "overall",
    "specific", "general", "paperwork", "obtained", "presented",
    "discussed", "ullah", "shahzad"
}

GENERIC_PHRASES = {
    "research paper", "research study", "present study",
    "current study", "related work", "future research",
    "previous research", "experimental results",
    "significant results", "different methods",
    "proposed method", "proposed approach", "various methods",
    "data analysis", "statistical analysis", "overall results",
    "research findings", "study findings"
}


def extract_keywords(text, top_n=15):
    """Extract frequent meaningful individual terms."""

    if not isinstance(text, str) or not text.strip():
        return []

    words = re.findall(r"\b[a-zA-Z][a-zA-Z-]{2,}\b", text.lower())

    filtered = [
        word.strip("-")
        for word in words
        if word not in STOP_WORDS
        and len(word.strip("-")) >= 3
    ]

    return Counter(filtered).most_common(top_n)


def _looks_like_person_name(phrase, original_text):
    """
    Filter likely personal names.

    This is a conservative heuristic: it checks whether the
    phrase occurs as two or more capitalized name-like words
    in the original passage.
    """

    words = _phrase_tokens(phrase)

    if len(words) < 2:
        return False

    escaped = r"\s+".join(
        re.escape(word) for word in words
    )

    pattern = (
        r"(?<![a-zA-Z])"
        + escaped
        + r"(?![a-zA-Z])"
    )

    matches = re.finditer(
        pattern,
        original_text,
        flags=re.IGNORECASE
    )

    for match in matches:
        original_phrase = match.group(0)

        # Both words capitalized is a likely name signal.
        capitalized_words = re.findall(
            r"[A-Z][a-z]{2,}",
            original_phrase
        )

        if len(capitalized_words) >= 2:
            return True

    return False


def _extract_phrases(text, max_words=3):
    """
    Extract meaningful two- and three-word concepts.

    Hyphenated words are split consistently so that phrase
    matching works for both extraction and evidence lookup.
    """

    if not isinstance(text, str) or not text.strip():
        return []

    words = re.findall(
        r"\b[a-zA-Z]{3,}\b",
        text.lower()
    )

    phrases = []

    for size in range(2, max_words + 1):
        for index in range(len(words) - size + 1):

            group = words[index:index + size]

            if any(word in STOP_WORDS for word in group):
                continue

            phrase = " ".join(group)

            if phrase in GENERIC_PHRASES:
                continue

            if _looks_like_person_name(phrase, text):
                continue

            phrases.append(phrase)

    return phrases


def _normalise_text(text):
    """Normalize text for duplicate-content comparisons."""

    if not isinstance(text, str):
        return ""

    return " ".join(
        re.findall(r"[a-z0-9]+", text.lower())
    )


def _phrase_tokens(phrase):
    """Convert a phrase into normalized tokens."""

    return re.findall(
        r"[a-z0-9]+",
        str(phrase).lower()
    )


def _is_subphrase(short_phrase, long_phrase):
    """Check whether one phrase occurs contiguously in another."""

    short_tokens = _phrase_tokens(short_phrase)
    long_tokens = _phrase_tokens(long_phrase)

    if not short_tokens or len(short_tokens) >= len(long_tokens):
        return False

    size = len(short_tokens)

    for index in range(len(long_tokens) - size + 1):
        if long_tokens[index:index + size] == short_tokens:
            return True

    return False


def _phrase_in_text(phrase, text):
    """Check whether a complete normalized phrase occurs in text."""

    phrase_tokens = _phrase_tokens(phrase)
    text_tokens = _phrase_tokens(text)

    if not phrase_tokens or len(phrase_tokens) > len(text_tokens):
        return False

    size = len(phrase_tokens)

    for index in range(len(text_tokens) - size + 1):
        if text_tokens[index:index + size] == phrase_tokens:
            return True

    return False


def _overlap_ratio(phrase_1, phrase_2):
    """Measure token overlap between two phrases."""

    tokens_1 = set(_phrase_tokens(phrase_1))
    tokens_2 = set(_phrase_tokens(phrase_2))

    if not tokens_1 or not tokens_2:
        return 0

    return len(tokens_1 & tokens_2) / min(
        len(tokens_1),
        len(tokens_2)
    )


# ============================================================
# 2. PASSAGE VALIDATION AND HELPERS
# ============================================================

ADMIN_HEADINGS = (
    "author contributions",
    "availability of data",
    "availability of materials",
    "competing interests",
    "conflict of interest",
    "funding",
    "acknowledgements",
    "acknowledgments",
    "consent for publication",
    "ethics approval",
    "declarations",
    "supplementary information",
    "supplementary materials"
)

ADMIN_PATTERNS = (
    r"no datasets? (?:were|was) generated",
    r"no datasets? (?:were|was) analysed",
    r"no datasets? (?:were|was) analyzed",
    r"data (?:are|is) available upon request",
    r"available from the corresponding author",
    r"the authors declare no competing interests",
    r"the authors declare that they have no",
    r"all authors read and approved the manuscript",
    r"this research received no external funding",
    r"not applicable",
    r"consent for publication",
    r"ethics approval and consent to participate"
)


def _valid_passages(passages):
    """Return passages with usable text and paper metadata."""

    valid = []

    for passage in passages or []:

        if not isinstance(passage, dict):
            continue

        text = passage.get("text", "")
        paper = passage.get("paper")

        if not isinstance(text, str) or not text.strip():
            continue

        if not paper:
            continue

        valid.append(passage)

    return valid


def _is_boilerplate(passage):
    """Identify publication declarations and administrative text."""

    heading = str(
        passage.get("heading", "")
    ).lower().strip()

    normalized_heading = _normalise_text(heading)

    if any(
        phrase in normalized_heading
        for phrase in ADMIN_HEADINGS
    ):
        return True

    text = passage.get("text", "").lower()

    for pattern in ADMIN_PATTERNS:
        if re.search(pattern, text):
            return True

    return False


def _section_evidence(passages, categories, limit=3):
    """Collect evidence from selected section categories."""

    selected = []

    for passage in passages:

        section = str(
            passage.get("section", "other")
        ).lower()

        if section in categories:
            selected.append({
                "text": passage["text"],
                "page": passage.get("page"),
                "heading": passage.get("heading", ""),
                "section": section,
                "paper": passage.get("paper")
            })

    return selected[:limit]


def _passage_keywords(passages, top_n=20):
    """Extract keywords from a collection of passages."""

    combined = " ".join(
        passage["text"]
        for passage in passages
    )

    return extract_keywords(combined, top_n=top_n)


def _make_concept(text_1, text_2):
    """Find shared meaningful terms between two passages."""

    words_1 = {
        word for word, _ in extract_keywords(text_1, top_n=30)
    }

    words_2 = {
        word for word, _ in extract_keywords(text_2, top_n=30)
    }

    shared = sorted(words_1 & words_2)

    if shared:
        return ", ".join(shared[:3])

    return "Semantically related research"


# ============================================================
# 3. SEMANTIC RESEARCH CONNECTIONS
# ============================================================

def analyze_connections(passages, top_n=15):
    """Find semantically related passages across different papers."""

    valid = _valid_passages(passages)

    valid = [
        passage
        for passage in valid
        if not _is_boilerplate(passage)
        and len(passage["text"].split()) >= 25
    ]

    if len(valid) < 2:
        return []

    papers = defaultdict(list)

    for index, passage in enumerate(valid):
        papers[passage["paper"]].append(index)

    if len(papers) < 2:
        return []

    model = get_model()

    texts = [
        passage["text"]
        for passage in valid
    ]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    similarity_matrix = cosine_similarity(embeddings)

    normalized_texts = [
        _normalise_text(text)
        for text in texts
    ]

    connections = []

    for paper_1, paper_2 in combinations(papers.keys(), 2):

        pair_connections = []

        for index_1 in papers[paper_1]:
            for index_2 in papers[paper_2]:

                passage_1 = valid[index_1]
                passage_2 = valid[index_2]

                text_1 = normalized_texts[index_1]
                text_2 = normalized_texts[index_2]

                if text_1 == text_2:
                    continue

                score = float(
                    similarity_matrix[index_1, index_2]
                )

                if score < 0.50:
                    continue

                if score >= 0.985:

                    words_1 = set(text_1.split())
                    words_2 = set(text_2.split())
                    union = words_1 | words_2

                    if union:
                        overlap = len(words_1 & words_2) / len(union)

                        if overlap >= 0.90:
                            continue

                concept = _make_concept(
                    passage_1["text"],
                    passage_2["text"]
                )

                shared_keywords = (
                    {
                        word for word, _ in extract_keywords(
                            passage_1["text"], 30
                        )
                    }
                    &
                    {
                        word for word, _ in extract_keywords(
                            passage_2["text"], 30
                        )
                    }
                )

                pair_connections.append({
                    "paper_1": paper_1,
                    "paper_2": paper_2,
                    "keyword": concept,
                    "score": round(score * 100, 2),
                    "frequency": len(shared_keywords),
                    "evidence_1": passage_1["text"],
                    "evidence_2": passage_2["text"],
                    "page_1": passage_1.get("page"),
                    "page_2": passage_2.get("page"),
                    "heading_1": passage_1.get("heading", ""),
                    "heading_2": passage_2.get("heading", ""),
                    "section_1": passage_1.get("section", "other"),
                    "section_2": passage_2.get("section", "other")
                })

        pair_connections.sort(
            key=lambda item: item["score"],
            reverse=True
        )

        seen_pairs = set()
        selected = []

        for connection in pair_connections:

            key = (
                _normalise_text(connection["evidence_1"]),
                _normalise_text(connection["evidence_2"])
            )

            if key in seen_pairs:
                continue

            seen_pairs.add(key)
            selected.append(connection)

            if len(selected) == 3:
                break

        connections.extend(selected)

    connections.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    return connections[:max(0, top_n)]


# ============================================================
# 4. PAPER COMPARISON ENGINE
# ============================================================

def compare_papers(passages, paper_names):
    """Compare two research papers using their extracted sections."""

    if len(paper_names) != 2:
        raise ValueError(
            "Exactly two paper names are required."
        )

    valid = _valid_passages(passages)

    comparison = {}

    section_map = {
        "objectives": {"abstract", "introduction"},
        "methodology": {"methodology", "methods"},
        "findings": {"results", "discussion"},
        "limitations": {"limitations"},
        "future_work": {"future_work"}
    }

    for paper_name in paper_names:

        paper_passages = [
            passage
            for passage in valid
            if passage["paper"] == paper_name
        ]

        paper_text = " ".join(
            passage["text"]
            for passage in paper_passages
        )

        section_results = {}

        for label, categories in section_map.items():

            section_results[label] = _section_evidence(
                paper_passages,
                categories,
                limit=3
            )

        comparison[paper_name] = {
            "keywords": extract_keywords(paper_text, top_n=10),
            "passage_count": len(paper_passages),
            "available": bool(paper_passages),
            "sections": section_results,
            "objectives": section_results["objectives"],
            "methodology": section_results["methodology"],
            "findings": section_results["findings"],
            "limitations": section_results["limitations"],
            "future_work": section_results["future_work"],
            "pages": sorted({
                passage.get("page")
                for passage in paper_passages
                if passage.get("page") is not None
            })
        }

    return comparison


# ============================================================
# 5. RESEARCH GAP EXPLORER
# ============================================================

def explore_research_gaps(passages, top_n=10):
    """
    Identify underrepresented multi-word concepts.

    Removes nested and highly overlapping phrases when they
    have the same paper coverage.

    These are collection-level observations, not verified
    gaps across all published literature.
    """

    valid = _valid_passages(passages)

    if not valid:
        return []

    paper_names = sorted({
        passage["paper"]
        for passage in valid
    })

    total_papers = len(paper_names)

    if total_papers < 2:
        return []

    paper_passages = defaultdict(list)

    for passage in valid:
        paper_passages[passage["paper"]].append(passage)

    topic_papers = defaultdict(set)
    topic_frequency = Counter()
    topic_evidence = defaultdict(list)

    for paper_name, paper_items in paper_passages.items():

        paper_topic_counts = Counter()

        for passage in paper_items:

            phrases = set(
                _extract_phrases(passage["text"])
            )

            for phrase in phrases:
                paper_topic_counts[phrase] += 1

                if len(topic_evidence[phrase]) < 5:
                    topic_evidence[phrase].append({
                        "paper": paper_name,
                        "page": passage.get("page"),
                        "heading": passage.get("heading", ""),
                        "section": passage.get("section", "other"),
                        "text": passage["text"]
                    })

        for topic, frequency in paper_topic_counts.items():

            topic_papers[topic].add(paper_name)
            topic_frequency[topic] += frequency

    candidates = []

    for topic, papers in topic_papers.items():

        paper_count = len(papers)

        if paper_count >= total_papers:
            continue

        if topic_frequency[topic] < 2:
            continue

        gap_score = 1 - (paper_count / total_papers)

        candidates.append({
            "topic": topic,
            "paper_count": paper_count,
            "total_papers": total_papers,
            "frequency": topic_frequency[topic],
            "gap_score": round(gap_score, 3),
            "papers": sorted(papers),
            "evidence": topic_evidence[topic][:5],
            "interpretation": (
                "This concept appears in fewer papers than "
                "others in the uploaded collection. This may "
                "indicate an area worth investigating, but "
                "additional literature review is required."
            )
        })

    candidates.sort(
        key=lambda item: (
            item["gap_score"],
            len(_phrase_tokens(item["topic"])),
            item["frequency"]
        ),
        reverse=True
    )

    filtered_gaps = []

    for candidate in candidates:

        candidate_papers = set(candidate["papers"])
        candidate_topic = candidate["topic"]

        duplicate = False

        for existing in filtered_gaps:

            if candidate_papers != set(existing["papers"]):
                continue

            existing_topic = existing["topic"]

            # Remove nested fragments.
            if (
                _is_subphrase(candidate_topic, existing_topic)
                or _is_subphrase(existing_topic, candidate_topic)
            ):
                duplicate = True
                break

            # Remove near-duplicate phrases with substantial
            # token overlap and identical paper coverage.
            if _overlap_ratio(
                candidate_topic,
                existing_topic
            ) >= 0.75:
                duplicate = True
                break

        if not duplicate:
            filtered_gaps.append(candidate)

    return filtered_gaps[:max(0, top_n)]


# ============================================================
# 6. IDEA CATALYST
# ============================================================

def generate_research_ideas(gaps, passages, top_n=5):
    """
    Generate distinct exploratory research directions.

    Each idea must have actual supporting passages from the
    papers associated with its gap.

    Ideas are hypotheses for further investigation, not
    validated scientific discoveries.
    """

    valid = _valid_passages(passages)

    if not gaps or not valid:
        return []

    ideas = []
    used_topics = []

    for gap in gaps:

        if len(ideas) >= max(0, top_n):
            break

        topic = gap["topic"]
        relevant_papers = set(gap.get("papers", []))

        # Avoid repeating a closely related topic.
        if any(
            _is_subphrase(topic, previous)
            or _is_subphrase(previous, topic)
            or _overlap_ratio(topic, previous) >= 0.75
            for previous in used_topics
        ):
            continue

        # Evidence must come from the papers associated
        # with this particular gap.
        matching_passages = [
            passage
            for passage in valid
            if passage["paper"] in relevant_papers
            and _phrase_in_text(topic, passage["text"])
        ]

        # Never generate an idea without evidence.
        if not matching_passages:
            continue

        related_counts = Counter()

        supporting_evidence = []

        for passage in matching_passages:

            phrases = set(
                _extract_phrases(passage["text"])
            )

            for phrase in phrases:

                if phrase == topic:
                    continue

                if (
                    _is_subphrase(phrase, topic)
                    or _is_subphrase(topic, phrase)
                ):
                    continue

                if _overlap_ratio(phrase, topic) >= 0.5:
                    continue

                related_counts[phrase] += 1

            if len(supporting_evidence) < 5:
                supporting_evidence.append({
                    "paper": passage["paper"],
                    "page": passage.get("page"),
                    "heading": passage.get("heading", ""),
                    "section": passage.get("section", "other"),
                    "text": passage["text"]
                })

        related_topics = []

        for phrase, _ in related_counts.most_common():

            if any(
                _is_subphrase(phrase, selected)
                or _is_subphrase(selected, phrase)
                or _overlap_ratio(phrase, selected) >= 0.75
                for selected in related_topics
            ):
                continue

            related_topics.append(phrase)

            if len(related_topics) == 3:
                break

        limitation_evidence = [
            passage
            for passage in matching_passages
            if passage.get("section") == "limitations"
        ][:3]

        future_evidence = [
            passage
            for passage in matching_passages
            if passage.get("section") == "future_work"
        ][:3]

        if related_topics:

            idea_text = (
                f"Investigate {topic} in relation to "
                f"{', '.join(related_topics)}. "
                "Examine how these concepts interact, compare "
                "the approaches used in the available studies, "
                "and identify a focused research question."
            )

        else:

            idea_text = (
                f"Investigate {topic} by examining the available "
                "evidence, comparing existing approaches, and "
                "identifying a focused research question."
            )

        if limitation_evidence:

            limitation_text = " ".join(
                limitation_evidence[0]["text"].split()
            )[:250]

            idea_text += (
                f" A limitation worth examining is: "
                f"{limitation_text}"
            )

        elif future_evidence:

            future_text = " ".join(
                future_evidence[0]["text"].split()
            )[:250]

            idea_text += (
                f" A possible next step is suggested by "
                f"the paper's future-work discussion: "
                f"{future_text}"
            )

        ideas.append({
            "topic": topic,
            "idea": idea_text,
            "related_topics": related_topics,
            "basis": (
                f"Appeared in {gap['paper_count']} of "
                f"{gap['total_papers']} uploaded papers."
            ),
            "papers": gap.get("papers", []),
            "evidence": supporting_evidence,
            "limitations": limitation_evidence,
            "future_work": future_evidence,
            "interpretation": (
                "Exploratory research direction generated "
                "from the uploaded collection. Validate it "
                "against additional literature before treating "
                "it as a genuine research gap."
            )
        })

        used_topics.append(topic)

    return ideas
