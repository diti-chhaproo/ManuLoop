# -*- coding: utf-8 -*-
"""
Week 6 DFM rule baseline: run geometry_rules.check_part() (fillet radius +
hole diameter, gated by AAGNet's predicted class) over a random sample of
MFCAD++ test-split parts, and tally how often each rule is violated.

Usage:
    python dfm_baseline.py
"""
import argparse
import json
import random
import time
from pathlib import Path

from infer import load_model, load_norm_stat, load_feature_schema
from geometry_rules import check_part

DEFAULT_TEST_DIR = Path(r"C:\manuloop\MFCAD_dataset\MFCAD++_dataset\step\test")
DEFAULT_OUTPUT_DIR = Path(__file__).parent / "output" / "dfm_baseline"


def sample_files(test_dir, sample_size, seed):
    all_files = sorted(test_dir.glob("*.step"))  # sort first so the seed reproducibly
                                                   # picks the same sample regardless of
                                                   # filesystem iteration order
    rng = random.Random(seed)
    return rng.sample(all_files, min(sample_size, len(all_files)))


def run_baseline(test_dir=DEFAULT_TEST_DIR, sample_size=500, seed=42, device="cuda"):
    print(f"Loading model/stat/feature schema...")
    model = load_model()
    stat = load_norm_stat()
    schema = load_feature_schema()

    files = sample_files(test_dir, sample_size, seed)
    print(f"Sampled {len(files)} files from {test_dir} (seed={seed})")

    per_file_results = {}
    failures = []
    total_faces = 0
    rule_face_violations = {"fillet_radius": 0, "hole_diameter": 0}
    rule_part_violations = {"fillet_radius": set(), "hole_diameter": set()}
    example_violations = {"fillet_radius": [], "hole_diameter": []}

    RULE_NAME_TO_KEY = {
        "3/4 - fillet radius": "fillet_radius",
        "10 - minimum cored-hole diameter": "hole_diameter",
    }

    t_start = time.time()
    for i, f in enumerate(files):
        try:
            result = check_part(str(f), model, stat, schema, device=device)
        except Exception as e:
            error = f"{type(e).__name__}: {e}"
            failures.append({"file": f.name, "error": error})
            print(f"[{i+1}/{len(files)}] FAILED {f.name}: {error}", flush=True)
            continue

        per_file_results[f.name] = result
        total_faces += len(result)

        part_had_violation = {"fillet_radius": False, "hole_diameter": False}
        for face_id, info in result.items():
            for v in info["violations"]:
                key = RULE_NAME_TO_KEY[v["rule"]]
                rule_face_violations[key] += 1
                part_had_violation[key] = True
                if len(example_violations[key]) < 5:
                    example_violations[key].append({
                        "file": f.name,
                        "face_id": face_id,
                        "predicted_class": info["predicted_class"],
                        "measured": v["measured"],
                        "limit": v["limit"],
                        "violation": v["violation"],
                    })
        for key, had in part_had_violation.items():
            if had:
                rule_part_violations[key].add(f.name)

        if (i + 1) % 50 == 0:
            print(f"[{i+1}/{len(files)}] processed, {len(failures)} failures so far, "
                  f"{time.time() - t_start:.0f}s elapsed", flush=True)

    dt = time.time() - t_start

    summary = {
        "seed": seed,
        "sample_size_requested": sample_size,
        "sample_size_actual": len(files),
        "files_succeeded": len(per_file_results),
        "files_failed": len(failures),
        "failures": failures,
        "total_faces_checked": total_faces,
        "elapsed_seconds": round(dt, 1),
        "rules": {
            key: {
                "faces_violated": rule_face_violations[key],
                "parts_with_violation": len(rule_part_violations[key]),
                "parts_with_violation_pct": round(
                    100 * len(rule_part_violations[key]) / len(per_file_results), 2
                ) if per_file_results else 0,
            }
            for key in rule_face_violations
        },
        "example_violations": example_violations,
    }

    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    per_file_path = DEFAULT_OUTPUT_DIR / "dfm_baseline_per_file.json"
    summary_path = DEFAULT_OUTPUT_DIR / "dfm_baseline_summary.json"

    with open(per_file_path, "w") as fp:
        json.dump(per_file_results, fp, indent=2)
    with open(summary_path, "w") as fp:
        json.dump(summary, fp, indent=2)

    print("\n" + "=" * 70)
    print("DFM BASELINE SUMMARY")
    print("=" * 70)
    print(f"Seed: {seed}")
    print(f"Files sampled: {len(files)}  succeeded: {len(per_file_results)}  failed: {len(failures)}")
    print(f"Total faces checked: {total_faces}")
    print(f"Elapsed: {dt:.0f}s ({dt/max(len(per_file_results),1):.2f}s/file)")
    print()
    for key, stats in summary["rules"].items():
        print(f"  {key}: {stats['faces_violated']} face violations, "
              f"{stats['parts_with_violation']} parts affected "
              f"({stats['parts_with_violation_pct']}%)")
    if failures:
        print(f"\nFailures ({len(failures)}):")
        for f in failures[:10]:
            print(f"  {f['file']}: {f['error']}")
        if len(failures) > 10:
            print(f"  ... and {len(failures) - 10} more (see summary JSON)")
    print(f"\nPer-file results saved to: {per_file_path}")
    print(f"Summary saved to: {summary_path}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dir", type=str, default=str(DEFAULT_TEST_DIR))
    parser.add_argument("--sample_size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    run_baseline(Path(args.test_dir), args.sample_size, args.seed, args.device)
