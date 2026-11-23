# Legal RAG

RAG system over Italian, Estonian and Slovenian civil law (inheritance and divorce codes, past cases).
Task A: single ReAct agent. Task B: supervisor with 9 specialized agents.
Streamlit app with chat, JSON log and RAGAS evaluation.

## Requirements

- Python 3.11
- A Groq API key
- The dataset in `data/Contest_Data` (not included in the repository)

## Setup

```bash
git clone https://github.com/LucaPesacane/legal-rag.git
cd legal-rag
echo "GROQ_API_KEY=your_key" > .env
```

Copy the dataset folders (`Italy`, `Estonia`, `slovenia`) into `data/Contest_Data`.

## Run locally

```bash
python -m venv .venv
.venv\Scripts\activate        # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python scripts/build_indexes.py
streamlit run app.py
```

The indexes (9 FAISS DBs in `vector_store/`) can also be built from the Vector DB Builder page.

## Run with Docker

Build the indexes first (see above), so that `data/` and `vector_store/` are copied into the image.

```bash
docker build -t legal-rag .
docker run -p 8501:8501 --env-file .env legal-rag
```

The app is available at http://localhost:8501.

## Structure

- `backend/`: loading, indexes, search, Task A and Task B pipelines, logging, RAGAS
- `pages/`: configuration, vector DB builder, chat, evaluation
- `eval/`: test set and evaluation scripts, results in `eval/results/`
- `eda/`, `scripts/`, `tests/`: corpus analysis, command line tools, tests
- `docs/`: report, slides and flowcharts
