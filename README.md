# paper-citation-verification
Program to verify citations in research papers.

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage
```bash
python main.py path/to/paper.pdf --out report.json
```

Output JSON contains `verified` and `unverified` arrays with match scores and Crossref metadata.

## Notes
- Verification uses Crossref and a heuristic score on title/author/year.
- Reference parsing is best-effort and improves when PDFs include a clear "References" section.
