import argparse
import json
import os
import re
import sys
import xml.etree.ElementTree as ET
from difflib import SequenceMatcher

import requests
from PyPDF2 import PdfReader
from dotenv import load_dotenv


YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
HEADING_RE = re.compile(r"^\s*(references|bibliography|works cited)\s*$", re.I)
STOP_HEADINGS = {
    "appendix",
    "acknowledgments",
    "acknowledgements",
    "supplementary",
    "supplemental",
    "algorithm",
    "proof",
    "proofs",
}
STOP_LINE_RE = re.compile(r"^(algorithm|figure|table)\s+\d+", re.I)


def extract_text(pdf_path):
    reader = PdfReader(pdf_path)
    chunks = []
    for page in reader.pages:
        text = page.extract_text() or ""
        chunks.append(text)
    return "\n".join(chunks)


def find_references_section(text):
    lines = text.splitlines()
    start_idx = None
    for i, line in enumerate(lines):
        if HEADING_RE.match(line.strip()):
            start_idx = i + 1
            break
    if start_idx is None:
        return text

    collected = []
    for line in lines[start_idx:]:
        stripped = line.strip()
        if not stripped:
            collected.append("")
            continue
        heading = stripped.lower()
        if heading in STOP_HEADINGS:
            break
        if STOP_LINE_RE.match(stripped):
            break
        if len(stripped) <= 40 and stripped.isupper() and not YEAR_RE.search(stripped):
            break
        collected.append(line)
    return "\n".join(collected)


def deepseek_extract_citations(reference_text):
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("Missing DEEPSEEK_API_KEY environment variable.")

    prompt = (
        "Extract citations from the references section below. "
        "Return strict JSON only: an array of objects with keys "
        '"title" and "authors" (authors as a list of full names). '
        "No extra text.\n\n"
        f"{reference_text}"
    )

    payload = {
        "model": "deepseek-reasoner",
        "messages": [
            {"role": "system", "content": "Return strict JSON only."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    response = requests.post(
        "https://api.deepseek.com/v1/chat/completions",
        headers=headers,
        json=payload
    )
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    return json.loads(content)


def normalize(text):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text.lower())).strip()


def semantic_scholar_lookup(title, authors):
    query_parts = [title] if title else []
    # if authors:
    #    query_parts.append(authors[0])
    query = " ".join(query_parts).strip()
    if not query:
        return None

    params = {
        "query": query,
        "limit": 1,
        "fields": "title,authors",
    }
    try:
        response = requests.get(
            "https://api.semanticscholar.org/graph/v1/paper/search/match",
            params=params
        )
        response.raise_for_status()
    except requests.RequestException:
        return None

    data = response.json()
    results = data.get("data", [])
    return results[0] if results else None


def extract_semantic_scholar_fields(item):
    if not item:
        return None, []
    title = item.get("title")
    authors = []
    for author in item.get("authors", []):
        name = author.get("name")
        if name:
            authors.append(name)
    return title, authors


def arxiv_lookup(title, authors):
    query_parts = [f'ti:"{title}"'] if title else []
    if authors:
        query_parts.append(f'au:"{authors[0]}"')
    if not query_parts:
        return None
    params = {
        "search_query": " AND ".join(query_parts),
        "start": 0,
        "max_results": 1,
    }
    try:
        response = requests.get(
            "https://export.arxiv.org/api/query",
            params=params,
            timeout=20,
        )
        response.raise_for_status()
    except requests.RequestException:
        return None

    try:
        root = ET.fromstring(response.text)
    except ET.ParseError:
        return None

    ns = {"atom": "http://www.w3.org/2005/Atom"}
    return root.find("atom:entry", ns)


def extract_arxiv_fields(entry):
    if entry is None:
        return None, []
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    title_node = entry.find("atom:title", ns)
    title = title_node.text.strip() if title_node is not None and title_node.text else None

    authors = []
    for author in entry.findall("atom:author", ns):
        name_node = author.find("atom:name", ns)
        if name_node is not None and name_node.text:
            authors.append(name_node.text.strip())
    return title, authors


def score_match_fields(ref, title, authors):
    title_score = 0.0
    if ref.get("title") and title:
        title_score = SequenceMatcher(
            None,
            normalize(ref["title"]),
            normalize(title),
        ).ratio()

    author_score = 0.0
    if ref.get("authors") and authors:
        ref_set = {normalize(a) for a in ref["authors"] if a}
        cr_set = {normalize(a) for a in authors if a}
        if ref_set:
            author_score = len(ref_set & cr_set) / len(ref_set)

    return title_score * 0.7 + author_score * 0.3


def build_report(references, min_score):
    verified = []
    unverified = []
    for ref in references:
        title = ref.get("title")
        authors = ref.get("authors", [])
        ss_item = semantic_scholar_lookup(title, authors)
        ss_title, ss_authors = extract_semantic_scholar_fields(ss_item)
        ss_score = score_match_fields(ref, ss_title, ss_authors) if ss_item else 0.0

        arxiv_entry = arxiv_lookup(title, authors)
        arxiv_title, arxiv_authors = extract_arxiv_fields(arxiv_entry)
        arxiv_score = score_match_fields(ref, arxiv_title, arxiv_authors) if arxiv_entry else 0.0

        score = max(ss_score, arxiv_score)
        entry = {
            "title": title,
            "authors": authors,
            "semantic_scholar": {
                "title": ss_title,
                "authors": ss_authors,
                "score": ss_score,
            } if ss_item is not None else None,
            "arxiv": {
                "title": arxiv_title,
                "authors": arxiv_authors,
                "score": arxiv_score,
            } if arxiv_entry is not None else None,
            "score": score,
        }
        if score >= min_score:
            verified.append(entry)
        else:
            unverified.append(entry)
    return {"verified": verified, "unverified": unverified}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Verify citations in a PDF against Semantic Scholar and arXiv."
    )
    parser.add_argument("pdf", help="Path to the PDF to analyze.")
    parser.add_argument("--out", default="citation_report.json", help="Output JSON path.")
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.5,
        help="Minimum score for verification (0-1).",
    )
    return parser.parse_args()


def main():
    load_dotenv()

    args = parse_args()
    text = extract_text(args.pdf)
    ref_section = find_references_section(text)
    citations = deepseek_extract_citations(ref_section)
    report = build_report(citations, args.min_score)

    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=True)

    print(f"Wrote report with {len(citations)} references to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
