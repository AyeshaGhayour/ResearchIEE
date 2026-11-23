
import os
import re
import math
import unicodedata
from collections import Counter, defaultdict

import pymupdf


# ==================================================
# 1. STANDARD RESEARCH SECTIONS
# ==================================================

STANDARD_SECTIONS = {
    "abstract": [
        "abstract", "summary"
    ],

    "introduction": [
        "introduction", "background",
        "motivation and key contributions",
        "novel contribution", "research objectives",
        "study objectives", "research questions"
    ],

    "literature_review": [
        "background and related work",
        "literature review", "related work",
        "related studies", "previous studies",
        "literature survey", "state of the art"
    ],

    "methodology": [
        "materials and methods",
        "research methodology",
        "research methods",
        "methodological framework",
        "methodological approach",
        "experimental methods",
        "experimental setup",
        "experimental design",
        "research design",
        "study design",
        "study selection process",
        "paper selection process",
        "literature search process",
        "data acquisition and cleaning",
        "data collection and analysis",
        "data collection",
        "data preprocessing",
        "preprocessing and feature engineering",
        "feature engineering",
        "model training",
        "training procedure",
        "transfer learning pipelines",
        "proposed methodology",
        "proposed method",
        "methodology",
        "methods",
        "method"
    ],

    "results": [
        "experimental results",
        "experimental evaluations",
        "empirical results",
        "empirical outcomes",
        "performance evaluation",
        "performance analysis",
        "evaluation results",
        "evaluation metrics",
        "comparative analysis",
        "quantitative results",
        "qualitative results",
        "findings and analysis",
        "main findings",
        "key findings",
        "results and analysis",
        "results"
    ],

    "discussion": [
        "results and discussion",
        "discussion of findings",
        "discussion"
    ],

    "limitations": [
        "scope and limitations",
        "study limitations",
        "research limitations",
        "limitations",
        "limitation"
    ],

    "future_work": [
        "future directions",
        "future research directions",
        "recommendations for future research",
        "future research",
        "future work"
    ],

    "conclusion": [
        "conclusions and recommendations",
        "conclusions",
        "conclusion"
    ],

    "references": [
        "bibliography",
        "references"
    ]
}


IGNORED_EXACT = {
    "open access",
    "survey",
    "review",
    "keywords",
    "declarations",
    "publisher's note",
    "publisher’s note",
    "acknowledgements",
    "acknowledgments",
    "supplementary information",
    "supplementary materials"
}


TABLE_LABELS = {
    "technique",
    "scalability",
    "interpretability",
    "accuracy",
    "efficiency",
    "precision",
    "recall",
    "f1-score",
    "specificity",
    "sensitivity",
    "objective",
    "main findings",
    "ml technique",
    "ref no",
    "ref. no",
    "references"
}


# ==================================================
# 2. TEXT CLEANING
# ==================================================

def clean_line(text):

    if not text:
        return ""

    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u00ad", "")

    text = "".join(
        char for char in text
        if unicodedata.category(char) != "Cc"
        or char in "\n\t\r"
    )

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def clean_page_text(text):

    if not text:
        return ""

    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u00ad", "")

    text = "".join(
        char for char in text
        if unicodedata.category(char) != "Cc"
        or char in "\n\t\r"
    )

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)

    return text.strip()


def normalize_heading(text):

    text = clean_line(text).lower()

    text = re.sub(
        r"^\s*\d+(?:\.\d+)*[.)]?\s*",
        "",
        text
    )

    return text.strip().rstrip(":").strip()


# ==================================================
# 3. SECTION CLASSIFICATION
# ==================================================

def classify_section(heading):

    normalized = normalize_heading(heading)

    if not normalized:
        return "other"

    aliases = []

    for section, names in STANDARD_SECTIONS.items():

        for name in names:
            aliases.append((name, section))

    aliases.sort(
        key=lambda item: len(item[0]),
        reverse=True
    )

    for name, section in aliases:

        if name == "method" and normalized != name:
            continue

        if re.match(
            rf"^{re.escape(name)}(?:\b|$)",
            normalized
        ):
            return section

    patterns = {
        "methodology": [
            r"^(research|study|experimental)\s+methodolog",
            r"^(data|dataset)\s+(collection|acquisition|preprocessing)",
            r"^(model|algorithm)\s+(training|development)",
            r"^(experimental|evaluation)\s+(setup|design)",
            r"^transfer learning pipeline",
            r"^feature engineering",
            r"^paper selection process"
        ],

        "results": [
            r"^(experimental|empirical|evaluation)\s+(result|outcome)",
            r"^(performance|comparative|quantitative|qualitative)\s+(analysis|evaluation)",
            r"^(key|main|major)\s+findings",
            r"^evaluation metrics"
        ],

        "discussion": [
            r"^interpretation of results",
            r"^discussion of findings"
        ],

        "limitations": [
            r"^(research|study|model)\s+limitations",
            r"^limitations and challenges"
        ],

        "future_work": [
            r"^future (work|research|directions)",
            r"^recommendations for future research"
        ],

        "introduction": [
            r"^(research|study)\s+(objectives|questions)",
            r"^motivation and contributions"
        ]
    }

    for section, expressions in patterns.items():

        for expression in expressions:

            if re.search(expression, normalized):
                return section

    return "other"


# ==================================================
# 4. PDF PAGE LINE EXTRACTION
# ==================================================

def extract_page_lines(page, page_number):

    page_data = page.get_text("dict")

    extracted = []

    for block in page_data["blocks"]:

        if "lines" not in block:
            continue

        for line in block["lines"]:

            spans = [
                span
                for span in line["spans"]
                if span["text"].strip()
            ]

            if not spans:
                continue

            text = " ".join(
                span["text"].strip()
                for span in spans
            )

            text = clean_line(text)

            if not text:
                continue

            extracted.append({

                "page": page_number,

                "text": text,

                "font_size": max(
                    span["size"]
                    for span in spans
                ),

                "bold": any(
                    span["flags"] & 16
                    for span in spans
                ),

                "y": line["bbox"][1],

                "bottom": line["bbox"][3],

                "x": line["bbox"][0],

                "page_height": page.rect.height,

                "line_index": len(extracted)
            })

    return extracted


# ==================================================
# 5. METADATA AND CAPTION FILTERING
# ==================================================

def is_metadata_or_caption(text):

    normalized = clean_line(text).lower()

    if not normalized:
        return True

    patterns = [
        r"^page\s+\d+(\s+of\s+\d+)?$",
        r"^vol\.?\s*\d+",
        r"\bdoi\s*:",
        r"doi\.org/",
        r"https?://",
        r"www\.",
        r"^©",
        r"^copyright\b",
        r"^received\s*:",
        r"^accepted\s*:",
        r"^published\s*:",
        r"^table\s*[\d.]+",
        r"^fig(?:ure)?\.?\s*[\d.]+",
        r"^algorithm\s+\d+",
        r"^keywords\s*:",
        r"^publisher'?s note$",
        r"\(\d{4}\)\s+\d+:\d+"
    ]

    if any(
        re.search(pattern, normalized)
        for pattern in patterns
    ):
        return True

    licence_indicators = [
        "creative commons",
        "open access",
        "non-commercial use",
        "noncommercial use",
        "appropriate credit",
        "reproduction in any medium",
        "distribution and reproduction",
        "licence terms",
        "license terms"
    ]

    licence_matches = sum(
        phrase in normalized
        for phrase in licence_indicators
    )

    if licence_matches >= 2:
        return True

    math_symbols = r"[=+−×÷∑√∫<>^_{}]"

    if len(re.findall(math_symbols, text)) >= 2:
        return True

    return False


# ==================================================
# 6. REPEATED HEADER AND FOOTER DETECTION
# ==================================================

def detect_repeated_headers(page_lines):

    total_pages = len(page_lines)

    if total_pages < 3:
        return set()

    occurrences = defaultdict(set)

    for page_number, lines in page_lines.items():

        for line in lines:

            page_height = line["page_height"]

            in_margin = (
                line["y"] < page_height * 0.12
                or line["bottom"] > page_height * 0.88
            )

            if not in_margin:
                continue

            text = clean_line(line["text"])

            if not text:
                continue

            if len(text) > 120:
                continue

            if len(text.split()) > 14:
                continue

            if is_metadata_or_caption(text):
                continue

            if classify_section(text) != "other":
                continue

            normalized = normalize_heading(text)

            if re.fullmatch(r"\d+", normalized):
                continue

            occurrences[normalized].add(page_number)

    minimum_occurrences = max(
        3,
        math.ceil(total_pages * 0.20)
    )

    return {
        text
        for text, pages in occurrences.items()
        if len(pages) >= minimum_occurrences
    }


# ==================================================
# 7. BODY FONT SIZE ESTIMATION
# ==================================================

def get_page_body_size(lines, repeated_headers=None):

    if repeated_headers is None:
        repeated_headers = set()

    sizes = []

    for line in lines:

        text = line["text"]

        normalized = normalize_heading(text)

        if normalized in repeated_headers:
            continue

        if is_metadata_or_caption(text):
            continue

        if len(text.split()) < 5:
            continue

        sizes.append(
            round(line["font_size"], 1)
        )

    if not sizes:
        return None

    return Counter(sizes).most_common(1)[0][0]


# ==================================================
# 8. HEADING DETECTION
# ==================================================

def is_likely_heading(
    line,
    body_size,
    repeated_headers=None
):

    if repeated_headers is None:
        repeated_headers = set()

    text = line["text"].strip()

    normalized = normalize_heading(text)

    if not text:
        return False

    if normalized in repeated_headers:
        return False

    if is_metadata_or_caption(text):
        return False

    if len(text) > 150:
        return False

    if len(text.split()) > 16:
        return False

    letters = sum(
        char.isalpha()
        for char in text
    )

    if letters < 3:
        return False

    if normalized in IGNORED_EXACT:
        return False

    if normalized.rstrip(":") in TABLE_LABELS:
        return False

    if re.match(
        r"^(ref\.?\s*no\.?|accuracy|f1[- ]?score)\s+"
        r"(objective|precision|recall|ml algorithm)",
        normalized
    ):
        return False

    if normalized.startswith("keywords"):
        return False

    if re.match(
        r"^(department|school|university|college|"
        r"faculty|institute)\b",
        normalized
    ):
        return False

    if re.search(
        r"\b(received|accepted|correspondence|"
        r"email|license|licence)\b",
        normalized
    ):
        return False

    if re.match(r"^[•●▪◦]", text):
        return False

    if re.match(
        r"^(and|or|but|which|that|because|therefore)\b",
        normalized
    ):
        return False

    if text.endswith((".", ";", ",")):
        return False

    numbered = bool(
        re.match(
            r"^\d+(?:\.\d+)*[.)]?\s+\S",
            text
        )
    )

    known_section = (
        classify_section(text) != "other"
    )

    if numbered:

        return (
            line["bold"]
            or (
                body_size is not None
                and line["font_size"] > body_size
            )
        )

    if known_section:

        if body_size is None:
            return True

        return (
            line["bold"]
            or line["font_size"] >= body_size
        )

    if body_size is None:
        return False

    larger_font = (
        line["font_size"] >= body_size + 0.25
    )

    if not line["bold"] and not larger_font:
        return False

    first_letter = next(
        (
            char
            for char in text
            if char.isalpha()
        ),
        ""
    )

    if first_letter and first_letter.islower():
        return False

    return True


# ==================================================
# 9. DOCUMENT HEADING DETECTION
# ==================================================

def detect_document_headings(
    page_lines,
    repeated_headers=None
):

    if repeated_headers is None:
        repeated_headers = set()

    headings = []

    for page_number, original_lines in page_lines.items():

        lines = original_lines

        if page_number == 1:

            start_index = next(
                (
                    index
                    for index, line in enumerate(lines)

                    if classify_section(line["text"])
                    in {
                        "abstract",
                        "introduction"
                    }
                ),
                0
            )

            lines = lines[start_index:]

        body_size = get_page_body_size(
            lines,
            repeated_headers
        )

        if body_size is None:
            continue

        for line in lines:

            if not is_likely_heading(
                line,
                body_size,
                repeated_headers
            ):
                continue

            text = line["text"].strip()

            numbered = re.match(
                r"^(\d+(?:\.\d+)*)(?:[.)])?\s+\S",
                text
            )

            level = 1

            if numbered:
                level = (
                    numbered.group(1).count(".") + 1
                )

            headings.append({

                "page": page_number,

                "line_index": line["line_index"],

                "heading": text,

                "level": level,

                "section": classify_section(text),

                "font_size": round(
                    line["font_size"],
                    1
                )
            })

    return headings


# ==================================================
# 10. SECTION SEGMENT EXTRACTION
# ==================================================

def build_section_segments(
    page_lines,
    headings,
    repeated_headers=None
):

    if repeated_headers is None:
        repeated_headers = set()

    heading_lookup = {

        (item["page"], item["line_index"]): item

        for item in headings
    }

    segments = []

    current_heading = "unclassified"

    current_section = "other"

    first_page_start = None

    for heading in headings:

        if (
            heading["page"] == 1
            and heading["section"] in {
                "abstract",
                "introduction"
            }
        ):

            first_page_start = heading["line_index"]

            break

    for page_number, lines in page_lines.items():

        current_text = []

        def flush_segment():

            if not current_text:
                return

            text = " ".join(current_text).strip()

            if text:

                segments.append({

                    "page": page_number,

                    "heading": current_heading,

                    "section": current_section,

                    "text": text
                })

            current_text.clear()

        for line in lines:

            if (
                page_number == 1
                and first_page_start is not None
                and line["line_index"] < first_page_start
            ):
                continue

            key = (
                page_number,
                line["line_index"]
            )

            detected_heading = heading_lookup.get(key)

            if detected_heading:

                flush_segment()

                current_heading = (
                    detected_heading["heading"]
                )

                current_section = (
                    detected_heading["section"]
                )

                continue

            normalized = normalize_heading(
                line["text"]
            )

            if normalized in repeated_headers:
                continue

            if is_metadata_or_caption(line["text"]):
                continue

            current_text.append(line["text"])

        flush_segment()

    return segments


# ==================================================
# 11. MAIN PDF READER
# ==================================================

def read_pdf(pdf_path):

    if not os.path.isfile(pdf_path):

        raise FileNotFoundError(
            f"PDF file not found: {pdf_path}"
        )

    try:

        with pymupdf.open(pdf_path) as document:

            total_pages = len(document)

            if total_pages == 0:
                raise ValueError(
                    "The PDF contains no pages."
                )

            pages = []

            page_lines = {}

            for page_number, page in enumerate(
                document,
                start=1
            ):

                text = clean_page_text(
                    page.get_text("text")
                )

                pages.append({

                    "page": page_number,

                    "text": text
                })

                page_lines[page_number] = (
                    extract_page_lines(
                        page,
                        page_number
                    )
                )

    except Exception as error:

        raise ValueError(
            f"Could not read PDF file: {error}"
        ) from error

    full_text = "\n\n".join(
        page["text"]
        for page in pages
    )

    repeated_headers = detect_repeated_headers(
        page_lines
    )

    headings = detect_document_headings(
        page_lines,
        repeated_headers
    )

    sections = build_section_segments(
        page_lines,
        headings,
        repeated_headers
    )

    return {

        "pages": pages,

        "full_text": full_text,

        "total_pages": total_pages,

        "headings": headings,

        "sections": sections
    }
