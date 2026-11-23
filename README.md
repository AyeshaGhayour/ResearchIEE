
# ResearchMind
### AI-Powered Research Intelligence Platform

**Don't just find research. Discover what connects it.**

ResearchMind is an AI-powered research intelligence platform designed to help researchers explore academic papers, discover relationships between studies, compare research methodologies, identify potential research gaps, and generate evidence-grounded research directions.

Built with Python, Flask, and Natural Language Processing, ResearchMind transforms uploaded research papers into an interactive research exploration workspace.

---

## Features

### 1. Semantic Research Explorer
- Search research papers using natural-language questions.
- Retrieve semantically relevant passages.
- View supporting paper names, page numbers, and relevance scores.

### 2. Research Connection Map
- Discover relationships between research papers.
- Identify shared concepts and research themes.
- Visualize connections through an interactive graph.

### 3. Paper Comparison Engine
- Compare multiple research papers.
- Explore research objectives and methodologies.
- Review datasets, findings, limitations, and future work.

### 4. Research Gap Explorer
- Identify potentially underrepresented topics in the uploaded collection.
- Analyze topic frequency and paper coverage.
- Present evidence-based research gap suggestions.

### 5. Idea Catalyst
- Generate potential research directions from identified gaps.
- Connect suggested ideas with supporting research evidence.
- Help researchers explore new research possibilities.

---

## Technology Stack

| Category | Technologies |
|---|---|
| Backend | Python, Flask |
| AI / NLP | Sentence Transformers |
| Semantic Search | NumPy, Scikit-learn |
| PDF Processing | PyMuPDF |
| Research Analysis | Python, NLP |
| Graph Analysis | NetworkX |
| Visualization | Plotly |
| Frontend | HTML, CSS, JavaScript |

---

## Project Structure

```text
ResearchMind/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── src/
│   ├── __init__.py
│   ├── pdf_reader.py
│   ├── text_processor.py
│   ├── search_engine.py
│   ├── research_analyzer.py
│   └── connection_graph.py
│
├── templates/
│   ├── index.html
│   ├── navbar.html
│   ├── search.html
│   ├── connections.html
│   ├── comparison.html
│   ├── gaps.html
│   └── ideas.html
│
├── static/
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── script.js
│


---

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/AyeshaGhayour/ResearchIEE.git
```

### 2. Navigate to the project

```bash
cd ResearchIEE
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

### 4. Activate the environment

**Windows PowerShell:**

```powershell
.venv\Scripts\Activate.ps1
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the application

```bash
python app.py
```

Open your browser and visit:

```text
http://127.0.0.1:5000
```

---

## How to Use

1. Launch ResearchMind.
2. Upload academic research papers in PDF format.
3. Search papers using natural-language research questions.
4. Explore connections between uploaded studies.
5. Compare research papers and their findings.
6. Discover potential research gaps within your collection.
7. Generate research ideas supported by relevant evidence.

---

## Testing

Run the feature test script:

```bash
python connection_test.py
```

The test script checks the five core ResearchMind features:

- Semantic Research Explorer
- Research Connection Map
- Paper Comparison Engine
- Research Gap Explorer
- Idea Catalyst

---

## Important Notes and Limitations

- ResearchMind analyzes the papers uploaded by the user; its findings are limited by the contents and quality of that collection.
- Suggested research gaps are potential underrepresented topics, not proof of gaps in the entire academic literature.
- Generated research ideas should be independently evaluated and validated.
- AI-generated analysis may require human review.
- The first model initialization may require an internet connection to obtain pretrained model files.

---

## Future Improvements

- Advanced research recommendation system.
- Improved research gap validation against broader literature.
- More sophisticated citation and reference extraction.
- Exportable research analysis reports.
- Enhanced paper summarization.
- Expanded visualization and comparison capabilities.

---

## Author

Developed as an AI and Machine Learning portfolio project.

**ResearchMind — Turning Research Papers into Research Intelligence.**
