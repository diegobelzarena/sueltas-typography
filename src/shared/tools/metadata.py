"""Corpus metadata helpers — skip-page filtering and CSV lookup utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_skip_pages(csv_path: Path | str) -> dict[str, set[str]]:
    """Load per-document page-skip sets from the corpus metadata CSV.

    The CSV must have a ``FileName`` column (document identifier) and a
    ``SkipPages`` column containing comma-separated **0-based** page
    indices.  For example, ``"2,3,4"`` means ``page_2.png``,
    ``page_3.png``, and ``page_4.png`` should be excluded.

    Parameters
    ----------
    csv_path : Path or str
        Path to the corpus CSV (e.g. ``ordered-table-corpus2.csv``).

    Returns
    -------
    dict[str, set[str]]
        Mapping ``{doc_name: {page_stems_to_skip}}``.
        Page stems follow the ``page_N`` convention (no extension).
        Documents with no pages to skip are omitted from the dict.
        Returns an empty dict if the file is missing, unreadable, or
        lacks the required columns.
    """
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        return {}

    try:
        df = pd.read_csv(csv_path, encoding="utf-8")
    except Exception:
        return {}

    if "FileName" not in df.columns or "SkipPages" not in df.columns:
        return {}

    skip_map: dict[str, set[str]] = {}
    for _, row in df.iterrows():
        doc_name = str(row["FileName"]).strip()
        raw = row.get("SkipPages")
        if pd.isna(raw) or str(raw).strip() == "":
            continue
        stems: set[str] = set()
        for token in str(raw).split(","):
            token = token.strip()
            if token.isdigit():
                stems.add(f"page_{token}")
        if stems:
            skip_map[doc_name] = stems

    return skip_map


def find_corpus_csv(corpus_dir: Path) -> Path | None:
    """Locate the metadata CSV inside a corpus directory.

    Prefers files matching ``*table*`` or ``*corpus*``, then falls back
    to the first ``.csv`` found.  Returns ``None`` when nothing matches.
    """
    corpus_dir = Path(corpus_dir)
    for pattern in ("*table*.csv", "*corpus*.csv", "*.csv"):
        matches = sorted(corpus_dir.glob(pattern))
        if matches:
            return matches[0]
    return None
