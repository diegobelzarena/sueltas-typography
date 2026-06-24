#!/usr/bin/env python
"""Master pipeline script for typographic analysis.

Runs the complete pipeline or selected steps:
  0. Source conversion (PDF/TIFF → PNG)
  1. CharNet / DocTR / Kraken OCR detection
  2. Character extraction via minimum cost paths
  3. Italic detection via structure tensor
  4. Unsupervised tree clustering
  5. Typographic distance computation
  6. A contrario analysis and visualisation
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Ensure the src/ packages are importable
# ---------------------------------------------------------------------------
_SRC = str(Path(__file__).resolve().parent.parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from shared.tools.metadata import find_corpus_csv
from shared.tools.report import StepReport


# ---------------------------------------------------------------------------
# Subprocess helper
# ---------------------------------------------------------------------------

def _run_cmd(cmd: list[str], cwd: Path) -> tuple[bool, str]:
    """Run a command, streaming stdout/stderr live with a prefix."""
    print(f"  Command: {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd, cwd=str(cwd),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    lines: list[str] = []
    for line in proc.stdout:
        stripped = line.rstrip()
        lines.append(stripped)
        print(f"    | {stripped}")
    proc.wait()
    if proc.returncode != 0:
        tail = "\n".join(lines[-30:]) if lines else "unknown error"
        return False, tail
    return True, ""


# ---------------------------------------------------------------------------
# Pipeline step definitions
# ---------------------------------------------------------------------------

STEPS = {
    0: {
        "name": "Source Conversion",
        "description": "Convert PDFs/TIFFs to normalized PNGs",
        "output_check": lambda p: p["imgs"].is_dir() and any(p["imgs"].iterdir()),
    },
    1: {
        "name": "OCR Detection + Recognition",
        "description": "Detect characters and words (CharNet, DocTR, or Kraken)", # Updated
        "output_check": lambda p: (p["ocr_dir"] / next(p["imgs"].iterdir()).name).exists()
            if any(p["imgs"].iterdir()) else False,
    },
    2: {
        "name": "Character Extraction",
        "description": "Segment characters via minimum cost paths",
        "output_check": lambda p: any(p["ocr_dir"].rglob("*_data.npz")),
    },
    3: {
        "name": "Italic Detection",
        "description": "Classify italic/round via structure tensor",
        "output_check": lambda p: any(p["ocr_dir"].rglob("italic_labels.npz")),
    },
    4: {
        "name": "Tree Clustering",
        "description": "Unsupervised GMM + tree clustering",
        "output_check": lambda p: any(p["ocr_dir"].rglob("clusters_all.npz")),
    },
    5: {
        "name": "Typographic Distances",
        "description": "Compute inter-document distances from cluster means",
        "output_check": lambda p: (p["corpus"] / "distances_roman.npz").exists() or
            (p["corpus"] / "distances_italic.npz").exists(),
    },
    6: {
        "name": "A Contrario Analysis",
        "description": "NFA-based detection and graph/matrix visualisation",
        "output_check": lambda p: (p["corpus"] / "results" / "acontrario_results.npz").exists(),
    },
}


# ---------------------------------------------------------------------------
# OCR Helpers & Runners
# ---------------------------------------------------------------------------

def _append_common_ocr_args(cmd: list[str], paths: dict, skip_existing: bool, 
                            metadata_csv: Path | None, single_doc: bool):
    """Append arguments common to all OCR detector scripts."""
    cmd.extend(["--report-dir", str(paths["reports_dir"])])
    if skip_existing:
        cmd.append("--skip-existing")
    if metadata_csv:
        cmd.extend(["--metadata-csv", str(metadata_csv)])
    if single_doc:
        cmd.append("--single-doc")


def find_charnet_config() -> str | None:
    """Locate the default CharNet config file."""
    script_dir = Path(__file__).parent
    candidates = [
        script_dir.parent / "configs" / "icdar2015_hourglass88.yaml",
        script_dir.parent / "src" / "charnet_src" / "configs" / "icdar2015_hourglass88.yaml",
        script_dir.parent / "src" / "charnet" / "configs" / "icdar2015_hourglass88.yaml",
    ]
    for cfg in candidates:
        if cfg.exists():
            return str(cfg)
    return None


def _run_step_1_charnet(paths, workers, skip_existing, config_file, single_doc, metadata_csv=None):
    """Run CharNet OCR."""
    if config_file is None:
        config_file = find_charnet_config()
        if config_file is None:
            return False, "Could not find CharNet config file"

    cmd = [
        sys.executable, "scripts/run_charnet.py",
        config_file, str(paths["imgs"]), str(paths["ocr_dir"]),
        "--workers", str(workers),
    ]
    _append_common_ocr_args(cmd, paths, skip_existing, metadata_csv, single_doc)
    return _run_cmd(cmd, paths["root"])


def _run_step_1_doctr(paths, skip_existing, single_doc, doctr_opts, metadata_csv=None):
    """Run DocTR detection + recognition."""
    cmd = [
        sys.executable, "scripts/run_doctr.py",
        str(paths["imgs"]), str(paths["ocr_dir"]),
        "--det-arch", doctr_opts.get("det_arch", "db_resnet50"),
    ]
    _append_common_ocr_args(cmd, paths, skip_existing, metadata_csv, single_doc)
    return _run_cmd(cmd, paths["root"])


def _run_step_1_kraken(paths, workers, skip_existing, single_doc, kraken_opts, metadata_csv=None):
    """Run Kraken OCR."""
    cmd = [
        sys.executable, "scripts/run_kraken.py",
        str(paths["imgs"]), str(paths["ocr_dir"]),
        "--workers", str(workers),
        "--rec-model", kraken_opts.get("rec_model", "catmus-print-fondue-large"),
    ]
    _append_common_ocr_args(cmd, paths, skip_existing, metadata_csv, single_doc)
    return _run_cmd(cmd, paths["root"])  # <--- FIXED: Added missing return


# ---------------------------------------------------------------------------
# Other Step Runners
# ---------------------------------------------------------------------------

def run_step_0(paths, workers, skip_existing, source_conv_opts=None, single_doc=False):
    if single_doc:
        return False, "Step 0 is not supported in single-doc mode"

    pdf_dir = paths.get("pdf_dir")
    if not pdf_dir or not Path(pdf_dir).is_dir():
        return False, f"PDF/source directory not found: {pdf_dir}"

    opts = source_conv_opts or {}
    cmd = [
        sys.executable, "scripts/convert_sources.py",
        str(pdf_dir), str(paths["imgs"]),
        "--target-dpi", str(opts.get("target_dpi", 150)),
        "--workers", str(workers),
        "--report-dir", str(paths["reports_dir"]),
    ]
    if opts.get("dpi_csv_dir"):
        cmd.extend(["--dpi-csv-dir", str(opts["dpi_csv_dir"])])
    if skip_existing:
        cmd.append("--skip-existing")

    return _run_cmd(cmd, paths["root"])


def run_step_1(paths, workers, skip_existing, config_file=None, single_doc=False,
               detector="charnet", doctr_opts=None, kraken_opts=None, metadata_csv=None):
    """Run OCR detection + recognition (CharNet, DocTR, or Kraken)."""
    if detector == "doctr":
        return _run_step_1_doctr(paths, skip_existing, single_doc, doctr_opts or {}, metadata_csv)
    elif detector == "kraken":
        return _run_step_1_kraken(paths, workers, skip_existing, single_doc, kraken_opts or {}, metadata_csv)
    
    return _run_step_1_charnet(paths, workers, skip_existing, config_file, single_doc, metadata_csv)


def run_step_2(paths, workers, skip_existing, single_doc=False, segmentation="box_init", metadata_csv=None):
    if segmentation == "kraken_init":
        cmd = [
            sys.executable, "scripts/character_extraction_kraken.py",
            str(paths["imgs"]), str(paths["ocr_dir"]),
            "--workers", str(workers), "--segmentation", segmentation,
            "--report-dir", str(paths["reports_dir"]),
        ]
    else:
        cmd = [
            sys.executable, "scripts/character_extraction.py",
            str(paths["imgs"]), str(paths["ocr_dir"]),
            "--workers", str(workers), "--segmentation", segmentation,
            "--report-dir", str(paths["reports_dir"]),
        ]
    if skip_existing: cmd.append("--skip-existing")
    if metadata_csv: cmd.extend(["--metadata-csv", str(metadata_csv)])
    if single_doc: cmd.append("--single-doc")
    return _run_cmd(cmd, paths["root"])


def run_step_3(paths, workers, skip_existing, single_doc=False):
    cmd = [
        sys.executable, "scripts/italic_detection.py", str(paths["ocr_dir"]),
        "--workers", str(workers), "--report-dir", str(paths["reports_dir"]),
    ]
    if skip_existing: cmd.append("--skip-existing")
    if single_doc: cmd.append("--single-doc")
    return _run_cmd(cmd, paths["root"])


def run_step_4(paths, workers, skip_existing, single_doc=False):
    cmd = [
        sys.executable, "scripts/clustering.py", str(paths["ocr_dir"]),
        "--workers", str(workers), "--report-dir", str(paths["reports_dir"]),
    ]
    if skip_existing: cmd.append("--skip-existing")
    if single_doc: cmd.append("--single-doc")
    return _run_cmd(cmd, paths["root"])


def run_step_5(paths, workers, skip_existing):
    cmd = [
        sys.executable, "scripts/typographic_distances.py", str(paths["corpus"]),
        "--ocr-dir", str(paths["ocr_dir"]), "--report-dir", str(paths["reports_dir"]),
    ]
    if skip_existing: cmd.append("--skip-existing")
    return _run_cmd(cmd, paths["root"])


def run_step_6(paths, workers, skip_existing, acontrario_config=None):
    if acontrario_config is None:
        corpus_name = paths["corpus"].name
        for suffix in [corpus_name, corpus_name.replace("-", "")]:
            candidate = paths["root"] / "configs" / f"acontrario_{suffix}.yaml"
            if candidate.exists():
                acontrario_config = str(candidate)
                break
        if acontrario_config is None:
            return False, "Could not find a contrario config file"

    cmd = [
        sys.executable, "scripts/run_acontrario.py", str(paths["corpus"]),
        "--config", acontrario_config, "--no-display", "--report-dir", str(paths["reports_dir"]),
    ]
    if skip_existing: cmd.append("--load-results")
    return _run_cmd(cmd, paths["root"])


STEP_RUNNERS = {
    0: run_step_0, 1: run_step_1, 2: run_step_2, 3: run_step_3,
    4: run_step_4, 5: run_step_5, 6: run_step_6,
}


# ---------------------------------------------------------------------------
# Path resolution & Validation (Unchanged from your original, kept for brevity)
# ---------------------------------------------------------------------------

def resolve_corpus_paths(input_path: Path, single_doc: bool = False,
                         detector: str = "charnet", ocr_dir_override: str | None = None):
    input_path = Path(input_path).resolve()

    if single_doc:
        corpus_root = input_path.parent.parent if input_path.parent.name == "imgs" else input_path.parent
        ocr_root = (Path(ocr_dir_override).resolve() if ocr_dir_override else _resolve_ocr_root(corpus_root, detector)) / input_path.name
        return {"root": corpus_root.parent.parent, "corpus": corpus_root, "imgs": input_path, "ocr_dir": ocr_root, "single_doc": True}

    if (input_path / "imgs").is_dir():
        corpus_root, imgs_root = input_path, input_path / "imgs"
    elif input_path.name == "imgs" and input_path.is_dir():
        corpus_root, imgs_root = input_path.parent, input_path
    else:
        corpus_root, imgs_root = input_path, input_path / "imgs"
        print(f"  Note: {imgs_root} does not exist yet (will be created by Step 0 if --pdf-dir is given)")

    ocr_root = Path(ocr_dir_override).resolve() if ocr_dir_override else _resolve_ocr_root(corpus_root, detector)
    return {"root": corpus_root.parent.parent, "corpus": corpus_root, "imgs": imgs_root, "ocr_dir": ocr_root, "single_doc": False}


def _resolve_ocr_root(corpus_root: Path, detector: str) -> Path:
    new_path = corpus_root / "ocr" / detector
    if new_path.is_dir(): return new_path
    legacy = corpus_root / "charnet"
    if legacy.is_dir() and not (corpus_root / "ocr").is_dir():
        print(f"  Note: using legacy charnet/ folder (migrate with 'mv charnet ocr/{detector}')")
        return legacy
    return new_path


def validate_paths(paths):
    errors = []
    if not paths["imgs"].is_dir():
        errors.append(f"Images directory not found: {paths['imgs']}")
    elif not any(paths["imgs"].iterdir()):
        errors.append(f"Images directory is empty: {paths['imgs']}")
    return errors


def print_status(paths, steps_to_run):
    print("\n" + "=" * 60 + "\nPipeline Status\n" + "=" * 60)
    print(f"\nCorpus:  {paths['corpus']}\nImages:  {paths['imgs']}\nOutput:  {paths['ocr_dir']}")
    cond = sum(1 for d in paths["imgs"].iterdir() if d.is_dir())
    print(f"\nMode:    {'Single document' if paths['single_doc'] else (f'Full corpus ({cond} documents)')}")

    print("\nSteps:")
    for step_num, step_info in STEPS.items():
        status = "[✓]" if paths["ocr_dir"].exists() and _safe_check(step_info["output_check"], paths) else "[ ]"
        marker = " <--" if step_num in steps_to_run else ""
        print(f"  {status} {step_num}. {step_info['name']}: {step_info['description']}{marker}")
    print()

def _safe_check(check_func, paths):
    try: return check_func(paths)
    except (StopIteration, OSError): return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run the complete typographic analysis pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n  python scripts/run_pipeline.py data/corpus-1 --steps 1,2,3,4\n"
    )
    parser.add_argument("input_dir", help="Corpus directory or document folder with --single-doc")
    parser.add_argument("--steps", default="1,2,3,4", help="Comma-separated list of steps (default: 1,2,3,4)")
    parser.add_argument("--single-doc", action="store_true", help="Process a single document folder")
    parser.add_argument("--workers", type=int, default=0, help="Number of parallel workers")
    parser.add_argument("--skip-existing", action="store_true", help="Skip documents/pages with existing outputs")
    parser.add_argument("--charnet-config", help="Path to CharNet config file")
    parser.add_argument("--acontrario-config", help="Path to a contrario YAML config")
    parser.add_argument("--dry-run", action="store_true", help="Show commands without executing")
    parser.add_argument("--pipeline-config", help="YAML config specifying detector/segmentation")
    parser.add_argument("--pdf-dir", help="Directory of PDFs/TIFFs for step 0")
    parser.add_argument("--csv", help="Metadata CSV to copy into corpus")
    parser.add_argument("--metadata-csv", help="Corpus CSV with a SkipPages column")
    parser.add_argument("--ocr-dir", help="Override the OCR output directory")
    parser.add_argument("--run-tag", help="Tag for reports directory")

    args = parser.parse_args(argv)

    # Load pipeline config
    pipeline_cfg = {}
    if args.pipeline_config:
        with open(args.pipeline_config, encoding="utf-8") as f:
            pipeline_cfg = yaml.safe_load(f) or {}

    detector = pipeline_cfg.get("detector", "charnet")
    segmentation = pipeline_cfg.get("segmentation", "box_init")
    
    # FIXED: Load detector-specific options from YAML
    doctr_opts = pipeline_cfg.get("doctr", {})
    kraken_opts = pipeline_cfg.get("kraken", {}) 
    source_conv_opts = pipeline_cfg.get("source_conversion", {})

    # Parse steps
    try:
        steps_to_run = [int(s.strip()) for s in args.steps.split(",")]
        for s in steps_to_run:
            if s not in STEPS: parser.error(f"Invalid step: {s}.")
    except ValueError:
        parser.error(f"Invalid steps format: {args.steps}.")

    if args.pdf_dir and 0 not in steps_to_run: steps_to_run.insert(0, 0)
    
    # Filter incompatible steps for single-doc
    for step in [0, 5, 6]:
        if args.single_doc and step in steps_to_run:
            print(f"Note: Step {step} skipped in single-doc mode")
            steps_to_run.remove(step)

    # Scaffold corpus if --pdf-dir
    corpus_dir = Path(args.input_dir).resolve()
    if args.pdf_dir:
        pdf_dir = Path(args.pdf_dir).resolve()
        if not pdf_dir.is_dir(): parser.error(f"PDF directory not found: {pdf_dir}")
        for sub in ("imgs", "results", "dpis", "reports"): (corpus_dir / sub).mkdir(parents=True, exist_ok=True)
        if args.csv:
            csv_src = Path(args.csv).resolve()
            if not csv_src.is_file(): parser.error(f"CSV file not found: {csv_src}")
            csv_dst = corpus_dir / csv_src.name
            if not csv_dst.exists(): shutil.copy2(str(csv_src), str(csv_dst))

    # Resolve paths
    paths = resolve_corpus_paths(args.input_dir, args.single_doc, detector=detector, ocr_dir_override=args.ocr_dir)
    if args.pdf_dir: paths["pdf_dir"] = str(pdf_dir)
    paths["ocr_dir"].mkdir(parents=True, exist_ok=True)

    # Metadata CSV
    metadata_csv = Path(args.metadata_csv).resolve() if args.metadata_csv else (find_corpus_csv(paths["corpus"]) if not args.single_doc else None)
    if metadata_csv and metadata_csv.is_file(): print(f"  Metadata CSV: {metadata_csv}")
    else: metadata_csv = None

    # Reports dir
    run_tag = args.run_tag or f"{detector}_{segmentation}_{time.strftime('%Y%m%d')}"
    reports_dir = paths["corpus"] / "reports" / run_tag
    reports_dir.mkdir(parents=True, exist_ok=True)
    paths["reports_dir"] = reports_dir
    paths["run_tag"] = run_tag

    if 0 not in steps_to_run:
        if errors := validate_paths(paths):
            for e in errors: print(f"ERROR: {e}", file=sys.stderr)
            return 1

    workers = args.workers or max(1, (os.cpu_count() or 2) - 1)
    print_status(paths, steps_to_run)

    if args.dry_run:
        print("Dry run - no commands executed.")
        return 0

    print("=" * 60 + "\nRunning Pipeline\n" + "=" * 60)
    total_start = time.time()

    for step_num in sorted(steps_to_run):
        step_info = STEPS[step_num]
        print(f"\n>> Step {step_num}: {step_info['name']}\n" + "-" * 40)
        step_start = time.time()
        runner = STEP_RUNNERS[step_num]

        # FIXED: Cleaned up step execution and passed kraken_opts
        if step_num == 0:
            success, error = runner(paths, workers, args.skip_existing, source_conv_opts, args.single_doc)
        elif step_num == 1:
            success, error = runner(
                paths, workers, args.skip_existing, args.charnet_config, args.single_doc,
                detector=detector, doctr_opts=doctr_opts, kraken_opts=kraken_opts, metadata_csv=metadata_csv
            )
        elif step_num == 2:
            success, error = runner(paths, workers, args.skip_existing, args.single_doc, segmentation=segmentation, metadata_csv=metadata_csv)
        elif step_num == 6:
            success, error = runner(paths, workers, args.skip_existing, args.acontrario_config)
        else:
            success, error = runner(paths, workers, args.skip_existing) if step_num == 5 else runner(paths, workers, args.skip_existing, args.single_doc)

        elapsed = time.time() - step_start
        if success:
            print(f"\n  ✓ Step {step_num} completed in {elapsed:.1f}s")
        else:
            print(f"\n  ✗ Step {step_num} failed: {error}\n\nPipeline aborted.")
            return 1

    total_elapsed = time.time() - total_start

    # Assemble combined pipeline report
    if reports_dir.is_dir():
        step_reports = {}
        for rfile in sorted(reports_dir.glob("*_report.json")):
            if rfile.name == "pipeline_report.json": continue
            try:
                with open(rfile, encoding="utf-8") as f: step_reports[rfile.stem] = json.load(f)
            except (json.JSONDecodeError, OSError): pass

        pipeline_report = {
            "pipeline": {
                "run_tag": run_tag, "steps_requested": sorted(steps_to_run),
                "detector": detector, "segmentation": segmentation,
                "ocr_dir": str(paths["ocr_dir"]), "workers": workers,
                "total_elapsed_s": round(total_elapsed, 2), "config_file": args.pipeline_config,
            },
            "step_reports": step_reports,
        }
        report_path = reports_dir / "pipeline_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(pipeline_report, f, indent=2, ensure_ascii=False)
        print(f"\nPipeline report saved → {report_path}")

    print("\n" + "=" * 60 + f"\nPipeline completed in {total_elapsed:.1f}s\n" + "=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())