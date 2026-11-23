import os
from src.text_processor import clean_text
import uuid
from flask import Flask, render_template, request, flash
from werkzeug.utils import secure_filename
from src.pdf_reader import read_pdf
from src.connection_graph import build_connection_graph
from src.research_analyzer import (
    analyze_connections,
    compare_papers,
    explore_research_gaps,
    generate_research_ideas
)
from src.search_engine import (
    prepare_paper_passages,
    build_search_index,
    search_papers
)


# =========================
# 1. Flask Configuration
# =========================

app = Flask(__name__)

app.secret_key = "researchmind-local-development-key"

UPLOAD_FOLDER = os.path.join(
    app.root_path,
    "data",
    "uploads"
)

ALLOWED_EXTENSIONS = {"pdf"}
MAX_FILE_SIZE = 20 * 1024 * 1024

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# =========================
# 2. Research Library
# =========================

uploaded_papers = []
all_passages = []
search_embeddings = None


# Research Gap Cache
gap_results_cache = None


def get_cached_gap_results():
    global gap_results_cache

    if gap_results_cache is None:
        print("Analyzing research gaps...")
        gap_results_cache = explore_research_gaps(
            all_passages,
            top_n=10
        )

    return gap_results_cache


def invalidate_gap_cache():
    global gap_results_cache
    gap_results_cache = None
    print("Research gap cache cleared.")


# =========================
# 3. Helper Functions
# =========================

def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )



def process_paper(file_path, original_name):
    """
    Extract PDF text, clean research sections,
    and prepare searchable passages.
    """

    result = read_pdf(file_path)

    if not result["full_text"].strip():
        raise ValueError("No extractable text found in this PDF.")

    if not result["sections"]:
        raise ValueError("No research sections could be extracted.")

    cleaned_sections = []

    for section in result["sections"]:

        cleaned_text = clean_text(
            section.get("text", "")
        )

        if not cleaned_text:
            continue

        cleaned_sections.append({
            "page": section.get("page"),
            "heading": section.get("heading", ""),
            "section": section.get("section", "other"),
            "text": cleaned_text
        })

    if not cleaned_sections:
        raise ValueError("No readable research sections found.")

    cleaned_full_text = "\n".join(
        section["text"] for section in cleaned_sections
    )

    cleaned_data = {
        "sections": cleaned_sections
    }

    new_passages = prepare_paper_passages(
        cleaned_data,
        original_name
    )

    if not new_passages:
        raise ValueError("No searchable passages found.")

    paper_info = {
        "name": original_name,
        "pages": result["total_pages"],
        "characters": len(cleaned_full_text),
        "passages": len(new_passages)
    }

    return paper_info, new_passages


# =========================
# 4. Restore Research Library
# =========================

def load_existing_library():
    """
    Restore previously uploaded PDFs when Flask starts.
    Rebuild the semantic search index.
    """

    global uploaded_papers
    global all_passages
    global search_embeddings

    print("\nRestoring ResearchMind library...")

    restored_papers = []
    restored_passages = []

    for filename in sorted(os.listdir(UPLOAD_FOLDER)):

        if not filename.lower().endswith(".pdf"):
            continue

        file_path = os.path.join(
            UPLOAD_FOLDER,
            filename
        )

        # Remove UUID prefix from stored filenames
        if len(filename) > 33 and filename[32] == "_":
            original_name = filename[33:]
        else:
            original_name = filename

        try:
            paper_info, paper_passages = process_paper(
                file_path,
                original_name
            )

            restored_papers.append(paper_info)
            restored_passages.extend(paper_passages)

            print(f"Restored: {original_name}")

        except Exception as error:
            print(f"Could not restore {filename}: {error}")

    uploaded_papers = restored_papers
    all_passages = restored_passages
    search_embeddings = None

    # Rebuild embeddings after restoring all papers
    if restored_passages:
        try:
            search_embeddings = build_search_index(
                restored_passages
            )

            print(
                f"Library restored: {len(uploaded_papers)} papers"
            )

        except Exception as error:
            print(f"Could not rebuild search index: {error}")

    else:
        print("No previously uploaded papers found.")

    print("ResearchMind is ready!\n")


# =========================
# 5. Home Route
# =========================

@app.route("/search", methods=["GET", "POST"])
def semantic_search():
    search_results = []
    question = ""

    if request.method == "POST":
        question = request.form.get("question", "").strip()

        if not question:
            flash("Please enter a research question.")

        elif not all_passages or search_embeddings is None:
            flash("Upload research papers before searching.")

        else:
            try:
                search_results = search_papers(
                    question,
                    all_passages,
                    search_embeddings,
                    top_k=5
                )

                if not search_results:
                    flash("No relevant passages were found.")

            except Exception as error:
                flash(f"Search failed: {error}")
                print(f"Search error: {error}")

    return render_template(
        "search.html",
        search_results=search_results,
        question=question,
        uploaded_papers=uploaded_papers
    )


@app.route("/connections")
def connections():
    connection_results = analyze_connections(
        all_passages,
        top_n=15
    )

    graph = build_connection_graph(connection_results)

    graph_html = graph.to_html(
        full_html=False,
        include_plotlyjs="cdn"
    )


    return render_template(
        "connections.html",
        graph_html=graph_html,
        connections=connection_results
    )

@app.route("/comparison", methods=["GET", "POST"])
def comparison():
    paper_names = sorted({
        passage.get("paper")
        for passage in all_passages
        if passage.get("paper")
    })

    comparison_results = None
    selected_papers = []

    if request.method == "POST":
        paper_1 = request.form.get("paper_1", "")
        paper_2 = request.form.get("paper_2", "")

        if (
            paper_1 in paper_names
            and paper_2 in paper_names
            and paper_1 != paper_2
        ):
            selected_papers = [paper_1, paper_2]

            comparison_results = compare_papers(
                all_passages,
                selected_papers
            )

    return render_template(
        "comparison.html",
        paper_names=paper_names,
        comparison_results=comparison_results,
        selected_papers=selected_papers
    )

@app.route("/gaps")
def gaps():
    gap_results = get_cached_gap_results()

    return render_template(
        "gaps.html",
        gap_results=gap_results
    )


@app.route("/ideas")
def ideas():
    gap_results = get_cached_gap_results()

    research_ideas = generate_research_ideas(
        gap_results,
        all_passages,
        top_n=5
    )

    return render_template(
        "ideas.html",
        research_ideas=research_ideas
    )

@app.route("/", methods=["GET", "POST"])
def home():

    global uploaded_papers
    global all_passages
    global search_embeddings

    search_results = []
    question = ""

    if request.method == "POST":

        # -------------------------
        # PDF Upload
        # -------------------------

        if "paper" in request.files:

            file = request.files["paper"]

            if file.filename == "":
                flash("Please select a PDF file.")

            elif not allowed_file(file.filename):
                flash("Only PDF files are allowed.")

            else:
                original_name = secure_filename(file.filename)

                if not original_name:
                    flash("Invalid filename.")

                else:
                    unique_name = (
                        f"{uuid.uuid4().hex}_{original_name}"
                    )

                    save_path = os.path.join(
                        app.config["UPLOAD_FOLDER"],
                        unique_name
                    )

                    file.save(save_path)

                    try:
                        paper_info, new_passages = process_paper(
                            save_path,
                            original_name
                        )

                        # Prepare a new index without changing
                        # the existing library until successful
                        candidate_passages = (
                            all_passages + new_passages
                        )

                        candidate_embeddings = build_search_index(
                            candidate_passages
                        )

                        if candidate_embeddings is None:
                            raise ValueError(
                                "Could not build the search index."
                            )

                        # Update library after successful indexing
                        all_passages = candidate_passages
                        search_embeddings = candidate_embeddings
                        uploaded_papers.append(paper_info)
                        invalidate_gap_cache()

                        flash(
                            f"{original_name} uploaded and indexed successfully!"
                        )

                    except Exception as error:

                        if os.path.exists(save_path):
                            os.remove(save_path)

                        flash(
                            f"PDF processing failed: {error}"
                        )

                        print(f"Upload error: {error}")

        # -------------------------
        # Semantic Research Search
        # -------------------------

        elif "question" in request.form:

            question = request.form.get(
                "question",
                ""
            ).strip()

            if not question:
                flash("Please enter a research question.")

            elif not all_passages or search_embeddings is None:
                flash(
                    "Upload research papers before searching."
                )

            else:
                try:
                    search_results = search_papers(
                        question,
                        all_passages,
                        search_embeddings,
                        top_k=5
                    )

                    if not search_results:
                        flash(
                            "No relevant passages were found."
                        )

                except Exception as error:
                    flash(f"Search failed: {error}")
                    print(f"Search error: {error}")

    return render_template(
        "index.html",
        uploaded_papers=uploaded_papers,
        search_results=search_results,
        question=question
    )


# =========================
# 6. File Size Error
# =========================

@app.errorhandler(413)
def file_too_large(error):

    flash("File is too large. Maximum allowed size is 20 MB.")

    return render_template(
        "index.html",
        uploaded_papers=uploaded_papers,
        search_results=[],
        question=""
    ), 413


# =========================
# 7. Start ResearchMind
# =========================

load_existing_library()


if __name__ == "__main__":

    app.run(
        debug=True,
        use_reloader=False
    )
