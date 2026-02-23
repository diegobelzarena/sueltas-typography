#!/usr/bin/env python
"""Master pipeline script for typographic analysis.

Runs the complete pipeline or selected steps:
  1. CharNet OCR detection
  2. Character extraction via minimum cost paths
  3. Italic detection via structure tensor
  4. Unsupervised tree clustering

Usage
-----
    # Run full pipeline on a corpus
    python scripts/run_pipeline.py data/corpus-1 --steps 1,2,3,4

    # Run only clustering (assumes previous steps completed)
    python scripts/run_pipeline.py data/corpus-1 --steps 4

    # Run on a single document
    python scripts/run_pipeline.py data/corpus-1/imgs/doc001 --single-doc
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# Pipeline step definitions
# ---------------------------------------------------------------------------

STEPS = {
    1: {
        "name": "CharNet OCR",
        "description": "Detect characters and words using neural network",
        "output_check": lambda p: (p["charnet"] / next(p["imgs"].iterdir()).name).exists()
            if any(p["imgs"].iterdir()) else False,
    },
    2: {
        "name": "Character Extraction",
        "description": "Segment characters via minimum cost paths",
        "output_check": lambda p: any(p["charnet"].rglob("*_data.npz")),
    },
    3: {
        "name": "Italic Detection",
        "description": "Classify italic/round via structure tensor",
        "output_check": lambda p: any(p["charnet"].rglob("italic_labels.npz")),
    },
    4: {
        "name": "Tree Clustering",
        "description": "Unsupervised GMM + tree clustering",
        "output_check": lambda p: any(p["charnet"].rglob("clusters_all.npz")),
    },
}


def find_charnet_config():
    """Locate the default CharNet config file."""
    script_dir = Path(__file__).parent
    candidates = [
        script_dir.parent / "src" / "charnet_src" / "configs" / "icdar2015_hourglass88.yaml",
        script_dir.parent / "src" / "charnet" / "configs" / "icdar2015_hourglass88.yaml",
        script_dir.parent / "configs" / "icdar2015_hourglass88.yaml",
    ]
    for cfg in candidates:
        if cfg.exists():
            return str(cfg)
    return None


def run_step_1(paths, workers, skip_existing, config_file=None):
    """Run CharNet OCR."""
    if config_file is None:
        config_file = find_charnet_config()
        if config_file is None:
            return False, "Could not find CharNet config file"

    cmd = [
        sys.executable, "scripts/run_charnet.py",
        config_file,
        str(paths["imgs"]),
        str(paths["charnet"]),
        "--workers", str(workers),
    ]
    if skip_existing:
        cmd.append("--skip-existing")

    print(f"  Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=paths["root"])
    return result.returncode == 0, ""


def run_step_2(paths, workers, skip_existing):
    """Run character extraction."""
    cmd = [
        sys.executable, "scripts/character_extraction.py",
        str(paths["imgs"]),
        str(paths["charnet"]),
        "--workers", str(workers),
    ]
    if skip_existing:
        cmd.append("--skip-existing")

    print(f"  Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=paths["root"])
    return result.returncode == 0, ""


def run_step_3(paths, workers, skip_existing):
    """Run italic detection."""
    cmd = [
        sys.executable, "scripts/italic_detection.py",
        str(paths["charnet"]),
        "--process-subfolders",
    ]
    if skip_existing:
        cmd.append("--skip-existing")

    print(f"  Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=paths["root"])
    return result.returncode == 0, ""


def run_step_4(paths, workers, skip_existing):
    """Run clustering."""
    cmd = [
        sys.executable, "scripts/clustering.py",
        str(paths["charnet"]),
        "--process-subfolders",
        "--workers", str(workers),
    ]
    if skip_existing:
        cmd.append("--skip-existing")

    print(f"  Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=paths["root"])
    return result.returncode == 0, ""


STEP_RUNNERS = {
    1: run_step_1,
    2: run_step_2,
    3: run_step_3,
    4: run_step_4,
}


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def resolve_corpus_paths(input_path: Path, single_doc: bool = False):
    """
    Resolve the standard corpus directory structure.
    
    For a corpus like data/corpus-1/:
        imgs/      -> document images
        charnet/   -> CharNet outputs + extraction outputs
    
    For single document mode, input_path points directly to a document folder.
    """
    input_path = Path(input_path).resolve()
    
    if single_doc:
        # input_path is the document folder itself (e.g., data/corpus-1/imgs/doc001)
        # Infer corpus root from imgs parent
        if input_path.parent.name == "imgs":
            corpus_root = input_path.parent.parent
        else:
            corpus_root = input_path.parent
        
        return {
            "root": corpus_root.parent,  # workspace root for subprocess cwd
            "corpus": corpus_root,
            "imgs": input_path,
            "charnet": corpus_root / "charnet" / input_path.name,
            "single_doc": True,
        }
    
    # Standard corpus mode
    # Check if input_path is the corpus root or imgs subfolder
    if (input_path / "imgs").is_dir():
        corpus_root = input_path
        imgs_root = input_path / "imgs"
    elif input_path.name == "imgs" and input_path.is_dir():
        corpus_root = input_path.parent
        imgs_root = input_path
    else:
        # Assume it's the corpus root without imgs/ subfolder
        corpus_root = input_path
        imgs_root = input_path
    
    # CharNet output goes to charnet/ sibling of imgs/
    charnet_root = corpus_root / "charnet"
    
    return {
        "root": corpus_root.parent,
        "corpus": corpus_root,
        "imgs": imgs_root,
        "charnet": charnet_root,
        "single_doc": False,
    }


def validate_paths(paths):
    """Check that required directories exist."""
    errors = []
    
    if not paths["imgs"].is_dir():
        errors.append(f"Images directory not found: {paths['imgs']}")
    elif not any(paths["imgs"].iterdir()):
        errors.append(f"Images directory is empty: {paths['imgs']}")
    
    return errors


# ---------------------------------------------------------------------------
# Status display
# ---------------------------------------------------------------------------

def print_status(paths, steps_to_run):
    """Print current pipeline status."""
    print("\n" + "=" * 60)
    print("Pipeline Status")
    print("=" * 60)
    print(f"\nCorpus:  {paths['corpus']}")
    print(f"Images:  {paths['imgs']}")
    print(f"Output:  {paths['charnet']}")
    
    if paths["single_doc"]:
        print(f"Mode:    Single document")
    else:
        n_docs = sum(1 for d in paths["imgs"].iterdir() if d.is_dir())
        print(f"Mode:    Full corpus ({n_docs} documents)")
    
    print("\nSteps:")
    for step_num, step_info in STEPS.items():
        status = "[ ]"
        if paths["charnet"].exists():
            try:
                if step_info["output_check"](paths):
                    status = "[✓]"
            except (StopIteration, OSError):
                pass
        
        marker = " <--" if step_num in steps_to_run else ""
        print(f"  {status} {step_num}. {step_info['name']}: {step_info['description']}{marker}")
    
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run the complete typographic analysis pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline
  python scripts/run_pipeline.py data/corpus-1 --steps 1,2,3,4

  # Run steps 2-4 (assumes CharNet already done)
  python scripts/run_pipeline.py data/corpus-1 --steps 2,3,4

  # Process single document
  python scripts/run_pipeline.py data/corpus-1/imgs/doc001 --single-doc --steps 1,2,3,4
        """
    )
    parser.add_argument(
        "input_dir",
        help="Corpus directory (containing imgs/) or document folder with --single-doc",
    )
    parser.add_argument(
        "--steps",
        default="1,2,3,4",
        help="Comma-separated list of steps to run (default: 1,2,3,4)",
    )
    parser.add_argument(
        "--single-doc",
        action="store_true",
        help="Process a single document folder instead of full corpus",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Number of parallel workers (default: ncpus-1)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip documents/pages that already have outputs",
    )
    parser.add_argument(
        "--charnet-config",
        help="Path to CharNet config file (auto-detected if not provided)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be run without executing",
    )

    args = parser.parse_args(argv)

    # Parse steps
    try:
        steps_to_run = [int(s.strip()) for s in args.steps.split(",")]
        for s in steps_to_run:
            if s not in STEPS:
                parser.error(f"Invalid step: {s}. Valid steps: 1,2,3,4")
    except ValueError:
        parser.error(f"Invalid steps format: {args.steps}. Use comma-separated numbers.")

    # Resolve paths
    paths = resolve_corpus_paths(args.input_dir, args.single_doc)
    
    # Validate
    errors = validate_paths(paths)
    if errors:
        for e in errors:
            print(f"ERROR: {e}", file=sys.stderr)
        return 1

    # Set workers
    workers = args.workers or max(1, (os.cpu_count() or 2) - 1)

    # Show status
    print_status(paths, steps_to_run)

    if args.dry_run:
        print("Dry run - no commands executed.")
        return 0

    # Run steps
    print("=" * 60)
    print("Running Pipeline")
    print("=" * 60)

    total_start = time.time()
    
    for step_num in sorted(steps_to_run):
        step_info = STEPS[step_num]
        print(f"\n>> Step {step_num}: {step_info['name']}")
        print("-" * 40)
        
        step_start = time.time()
        runner = STEP_RUNNERS[step_num]
        
        # Step 1 needs config file
        if step_num == 1:
            success, error = runner(paths, workers, args.skip_existing, args.charnet_config)
        else:
            success, error = runner(paths, workers, args.skip_existing)
        
        elapsed = time.time() - step_start
        
        if success:
            print(f"\n  ✓ Step {step_num} completed in {elapsed:.1f}s")
        else:
            print(f"\n  ✗ Step {step_num} failed: {error}")
            print("\nPipeline aborted.")
            return 1

    total_elapsed = time.time() - total_start
    print("\n" + "=" * 60)
    print(f"Pipeline completed in {total_elapsed:.1f}s")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
