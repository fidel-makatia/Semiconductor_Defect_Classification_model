"""Run all training and evaluation for both Problem A and Problem D.

Usage:
    python run_all.py              # Train + evaluate both problems
    python run_all.py --eval-only  # Skip training, just evaluate (requires existing checkpoints)
    python run_all.py --problem a  # Only Problem A
    python run_all.py --problem d  # Only Problem D
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run_script(script_path, label):
    """Run a Python script and stream output live."""
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"  Script: {script_path}")
    print(f"{'='*70}\n")

    t0 = time.time()
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(ROOT),
        # Stream output live to console
    )
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"\n  FAILED (exit code {result.returncode}) after {elapsed:.0f}s")
        return False
    else:
        print(f"\n  COMPLETED in {elapsed:.0f}s")
        return True


def main():
    parser = argparse.ArgumentParser(description="Run all training and evaluation")
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip training, only run evaluation")
    parser.add_argument("--problem", choices=["a", "d", "both"], default="both",
                        help="Which problem to run (default: both)")
    args = parser.parse_args()

    results = {}

    # =====================================================================
    # PROBLEM D: IR Drop Prediction
    # =====================================================================
    if args.problem in ("d", "both"):
        if not args.eval_only:
            ok = run_script(ROOT / "problem_d" / "train.py",
                            "PROBLEM D: Training Attention U-Net for IR Drop Prediction")
            results["D-train"] = ok
            if not ok:
                print("Problem D training failed! Skipping evaluation.")

        ckpt_d = ROOT / "problem_d" / "checkpoints" / "best_model.pt"
        if ckpt_d.exists() or args.eval_only:
            ok = run_script(ROOT / "problem_d" / "evaluate.py",
                            "PROBLEM D: Evaluating IR Drop Prediction")
            results["D-eval"] = ok
        else:
            print(f"No checkpoint found at {ckpt_d}, skipping Problem D evaluation")
            results["D-eval"] = False

    # =====================================================================
    # PROBLEM A: Few-Shot Defect Classification
    # =====================================================================
    if args.problem in ("a", "both"):
        if not args.eval_only:
            ok = run_script(ROOT / "problem_a" / "train.py",
                            "PROBLEM A: Training Prototypical Network for Defect Classification")
            results["A-train"] = ok
            if not ok:
                print("Problem A training failed! Skipping evaluation.")

        ckpt_a = ROOT / "problem_a" / "checkpoints" / "best_model.pt"
        if ckpt_a.exists() or args.eval_only:
            ok = run_script(ROOT / "problem_a" / "evaluate.py",
                            "PROBLEM A: Evaluating Few-Shot Defect Classification")
            results["A-eval"] = ok
        else:
            print(f"No checkpoint found at {ckpt_a}, skipping Problem A evaluation")
            results["A-eval"] = False

    # =====================================================================
    # SUMMARY
    # =====================================================================
    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")
    for step, ok in results.items():
        status = "OK" if ok else "FAILED"
        print(f"  {step:20s}: {status}")

    # List generated outputs
    print(f"\n{'='*70}")
    print("  GENERATED OUTPUTS")
    print(f"{'='*70}")

    for problem in ["problem_a", "problem_d"]:
        outputs_dir = ROOT / problem / "outputs"
        if outputs_dir.exists():
            pngs = sorted(outputs_dir.glob("*.png"))
            if pngs:
                print(f"\n  {problem}/outputs/:")
                for p in pngs:
                    size_kb = p.stat().st_size / 1024
                    print(f"    {p.name:45s} ({size_kb:.0f} KB)")

        ckpt_dir = ROOT / problem / "checkpoints"
        if ckpt_dir.exists():
            ckpts = sorted(ckpt_dir.glob("*.pt"))
            if ckpts:
                print(f"\n  {problem}/checkpoints/:")
                for c in ckpts:
                    size_mb = c.stat().st_size / (1024 * 1024)
                    print(f"    {c.name:45s} ({size_mb:.1f} MB)")

    print(f"\n{'='*70}")
    all_ok = all(results.values())
    if all_ok:
        print("  ALL STEPS COMPLETED SUCCESSFULLY!")
    else:
        failed = [k for k, v in results.items() if not v]
        print(f"  Some steps failed: {', '.join(failed)}")
    print(f"{'='*70}\n")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
