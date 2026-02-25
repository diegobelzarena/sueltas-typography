#!/usr/bin/env python
"""Validate pipeline outputs for a document.

Checks that all expected output files exist and have the correct structure.

Usage
-----
    python scripts/validate_outputs.py data/corpus-1/charnet/doc001
    python scripts/validate_outputs.py data/corpus-1/charnet --all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def validate_charnet_json(json_path):
    """Validate a CharNet JSON output file."""
    import json
    
    errors = []
    try:
        with open(json_path) as f:
            words = json.load(f)
        
        if not isinstance(words, list):
            errors.append("Root is not a list")
            return errors
        
        for i, word in enumerate(words):
            if "tblr" not in word:
                errors.append(f"Word {i}: missing 'tblr'")
            elif not isinstance(word["tblr"], list) or len(word["tblr"]) != 4:
                errors.append(f"Word {i}: 'tblr' should be [t, b, l, r]")
            
            if "chars" in word:
                for j, char in enumerate(word["chars"]):
                    if "tblr" not in char:
                        errors.append(f"Word {i}, char {j}: missing 'tblr'")
    
    except json.JSONDecodeError as e:
        errors.append(f"Invalid JSON: {e}")
    except Exception as e:
        errors.append(f"Error reading file: {e}")
    
    return errors


def validate_data_npz(npz_path):
    """Validate a character extraction .npz file."""
    errors = []
    required_keys = ["char_imgs", "char_labels", "char_word_idx", 
                     "word_orientations", "word_stroke_orientations"]
    
    try:
        data = np.load(str(npz_path), allow_pickle=True)
        
        for key in required_keys:
            if key not in data:
                errors.append(f"Missing key: '{key}'")
        
        if "char_imgs" in data:
            imgs = data["char_imgs"]
            if imgs.ndim != 3:
                errors.append(f"char_imgs should be 3D, got {imgs.ndim}D")
            elif imgs.shape[1:] != (40, 32):
                errors.append(f"char_imgs should be (N, 40, 32), got {imgs.shape}")
        
        # Check array lengths match
        if all(k in data for k in ["char_imgs", "char_labels", "char_word_idx"]):
            n_chars = len(data["char_imgs"])
            if len(data["char_labels"]) != n_chars:
                errors.append(f"char_labels length mismatch: {len(data['char_labels'])} vs {n_chars}")
            if len(data["char_word_idx"]) != n_chars:
                errors.append(f"char_word_idx length mismatch: {len(data['char_word_idx'])} vs {n_chars}")
    
    except Exception as e:
        errors.append(f"Error reading file: {e}")
    
    return errors


def validate_italic_npz(npz_path):
    """Validate an italic_labels.npz file."""
    errors = []
    required_keys = ["char_italic", "threshold", "page_names", "page_char_counts"]
    
    try:
        data = np.load(str(npz_path), allow_pickle=True)
        
        for key in required_keys:
            if key not in data:
                errors.append(f"Missing key: '{key}'")
        
        if "char_italic" in data:
            italic = data["char_italic"]
            if italic.dtype not in (bool, np.bool_, np.int8):
                errors.append(f"char_italic should be bool or int8, got {italic.dtype}")
    
    except Exception as e:
        errors.append(f"Error reading file: {e}")
    
    return errors


def validate_clusters_npz(npz_path):
    """Validate a clusters_all.npz file."""
    errors = []
    required_keys = ["cluster_labels", "cluster_means", "cluster_italic"]
    
    try:
        data = np.load(str(npz_path), allow_pickle=True)
        
        for key in required_keys:
            if key not in data:
                errors.append(f"Missing key: '{key}'")
        
        if "cluster_means" in data:
            means = data["cluster_means"]
            if means.ndim != 3:
                errors.append(f"cluster_means should be 3D, got {means.ndim}D")
            elif means.shape[1:] != (40, 32):
                errors.append(f"cluster_means should be (K, 40, 32), got {means.shape}")
        
        # Check cluster count consistency
        if all(k in data for k in ["cluster_means", "cluster_italic"]):
            n_clusters = len(data["cluster_means"])
            if len(data["cluster_italic"]) != n_clusters:
                errors.append(f"cluster_italic length mismatch")
    
    except Exception as e:
        errors.append(f"Error reading file: {e}")
    
    return errors


def validate_document(doc_dir):
    """Validate all outputs for a single document."""
    doc_dir = Path(doc_dir)
    results = {"document": doc_dir.name, "steps": {}}
    
    # Step 1: CharNet JSON files
    json_files = list(doc_dir.glob("*.json"))
    if not json_files:
        results["steps"]["1_charnet"] = {"status": "missing", "files": 0}
    else:
        errors = []
        for jf in json_files:
            file_errors = validate_charnet_json(jf)
            if file_errors:
                errors.extend([f"{jf.name}: {e}" for e in file_errors])
        results["steps"]["1_charnet"] = {
            "status": "ok" if not errors else "errors",
            "files": len(json_files),
            "errors": errors,
        }
    
    # Step 2: Character extraction
    data_files = list(doc_dir.glob("*_data.npz"))
    if not data_files:
        results["steps"]["2_extraction"] = {"status": "missing", "files": 0}
    else:
        errors = []
        for df in data_files:
            file_errors = validate_data_npz(df)
            if file_errors:
                errors.extend([f"{df.name}: {e}" for e in file_errors])
        results["steps"]["2_extraction"] = {
            "status": "ok" if not errors else "errors",
            "files": len(data_files),
            "errors": errors,
        }
    
    # Step 3: Italic detection
    italic_path = doc_dir / "italic_labels.npz"
    if not italic_path.exists():
        results["steps"]["3_italic"] = {"status": "missing"}
    else:
        errors = validate_italic_npz(italic_path)
        results["steps"]["3_italic"] = {
            "status": "ok" if not errors else "errors",
            "errors": errors,
        }
    
    # Step 4: Clustering
    cluster_path = doc_dir / "clusters_all.npz"
    if not cluster_path.exists():
        results["steps"]["4_clustering"] = {"status": "missing"}
    else:
        errors = validate_clusters_npz(cluster_path)
        results["steps"]["4_clustering"] = {
            "status": "ok" if not errors else "errors",
            "errors": errors,
        }
    
    return results


def print_results(results):
    """Pretty-print validation results."""
    print(f"\nDocument: {results['document']}")
    print("-" * 50)
    
    all_ok = True
    for step_name, step_info in results["steps"].items():
        status = step_info["status"]
        
        if status == "ok":
            symbol = "✓"
        elif status == "missing":
            symbol = "○"
            all_ok = False
        else:
            symbol = "✗"
            all_ok = False
        
        files_str = f" ({step_info.get('files', 0)} files)" if "files" in step_info else ""
        print(f"  [{symbol}] {step_name}{files_str}")
        
        if "errors" in step_info and step_info["errors"]:
            for err in step_info["errors"][:3]:  # Show first 3 errors
                print(f"        → {err}")
            if len(step_info["errors"]) > 3:
                print(f"        → ... and {len(step_info['errors']) - 3} more errors")
    
    return all_ok


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Validate pipeline outputs for documents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python scripts/validate_outputs.py data/corpus-1/charnet/doc001
  python scripts/validate_outputs.py data/corpus-1/charnet --all
        """,
    )
    parser.add_argument(
        "input_path",
        help="Document folder or parent folder with --all",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Validate all subdirectories as documents",
    )
    
    args = parser.parse_args(argv)
    input_path = Path(args.input_path)
    
    if not input_path.exists():
        print(f"ERROR: Path not found: {input_path}", file=sys.stderr)
        return 1
    
    if args.all:
        doc_dirs = [d for d in sorted(input_path.iterdir()) if d.is_dir()]
    else:
        doc_dirs = [input_path]
    
    if not doc_dirs:
        print("No documents found.")
        return 1
    
    all_ok = True
    for doc_dir in doc_dirs:
        results = validate_document(doc_dir)
        if not print_results(results):
            all_ok = False
    
    print("\n" + "=" * 50)
    if all_ok:
        print("All validations passed!")
    else:
        print("Some validations failed. Check errors above.")
    
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
