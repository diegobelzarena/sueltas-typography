"""Lightweight step-level reporting for the typographic analysis pipeline.

Each pipeline step collects statistics into a :class:`StepReport` and saves
it as a JSON file.  ``run_pipeline.py`` assembles individual step reports
into a combined ``pipeline_report.json``.

Usage inside a step script::

    from shared.tools.report import StepReport

    report = StepReport("ocr_detection", detector="doctr")
    report.add_document("BNE_1001", {"n_pages": 22, "n_words": 1234})
    report.set_summary({"total_documents": 1, "total_words": 1234})
    report.save(Path("data/corpus-1/reports"))
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path


class StepReport:
    """Accumulates statistics for a single pipeline step.

    Parameters
    ----------
    step_name : str
        Short identifier used as the JSON filename stem
        (e.g. ``"ocr_detection"``, ``"clustering"``).
    **extra
        Arbitrary key-value pairs stored under ``"config"`` in the report
        (detector name, segmentation mode, etc.).
    """

    def __init__(self, step_name: str, **extra) -> None:
        self.step_name = step_name
        self.config: dict = dict(extra) if extra else {}
        self.documents: dict[str, dict] = {}
        self.summary: dict = {}
        self._t0 = time.time()

    # ------------------------------------------------------------------
    # Accumulation helpers
    # ------------------------------------------------------------------

    def add_document(self, doc_name: str, info: dict) -> None:
        """Record per-document statistics."""
        self.documents[doc_name] = info

    def add_page(self, doc_name: str, page_name: str, info: dict) -> None:
        """Record per-page statistics nested under a document."""
        doc = self.documents.setdefault(doc_name, {"pages": {}})
        pages = doc.setdefault("pages", {})
        pages[page_name] = info

    def set_summary(self, summary: dict) -> None:
        """Set or replace the corpus-level summary section."""
        self.summary = summary

    def update_summary(self, **kwargs) -> None:
        """Merge key-value pairs into the summary."""
        self.summary.update(kwargs)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def elapsed(self) -> float:
        """Seconds since the report was created."""
        return time.time() - self._t0

    def to_dict(self) -> dict:
        """Serialise to a plain dict (JSON-safe)."""
        d: dict = {
            "step": self.step_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": round(self.elapsed(), 2),
        }
        if self.config:
            d["config"] = self.config
        if self.documents:
            d["documents"] = self.documents
        if self.summary:
            d["summary"] = self.summary
        return d

    def save(self, report_dir: str | Path) -> Path:
        """Write the report as JSON.

        Parameters
        ----------
        report_dir : Path
            Directory where ``{step_name}_report.json`` will be written.

        Returns
        -------
        Path to the written file.
        """
        report_dir = Path(report_dir)
        report_dir.mkdir(parents=True, exist_ok=True)
        out = report_dir / f"{self.step_name}_report.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False,
                      default=str)
        return out

    # ------------------------------------------------------------------
    # Loading / merging
    # ------------------------------------------------------------------

    @staticmethod
    def load(path: str | Path) -> dict:
        """Load a report JSON into a plain dict."""
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def merge_reports(report_dir: str | Path,
                      step_names: list[str] | None = None) -> dict:
        """Load and merge all ``*_report.json`` files in *report_dir*.

        Returns a dict keyed by step name.
        """
        report_dir = Path(report_dir)
        merged: dict = {}
        for p in sorted(report_dir.glob("*_report.json")):
            step = p.stem.replace("_report", "")
            if step_names and step not in step_names:
                continue
            merged[step] = StepReport.load(p)
        return merged
