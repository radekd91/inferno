"""Compare FLAME pickle outputs from run_emote_backend_diagnostic.py."""

import argparse
import pickle
from pathlib import Path

import numpy as np


ARRAY_KEYS = ("shape", "expression", "jaw_pose", "global_pose")


def load_case(case_dir):
    values = {key: [] for key in ARRAY_KEYS}
    files = sorted(case_dir.rglob("*.pkl"))
    if not files:
        return files, None
    for path in files:
        with path.open("rb") as file:
            data = pickle.load(file)
        for key in ARRAY_KEYS:
            values[key].append(np.asarray(data[key]).reshape(-1))
    return files, {key: np.concatenate(parts) for key, parts in values.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--baseline", default="vanilla_numeric")
    args = parser.parse_args()

    baseline_files, baseline = load_case(args.root / args.baseline)
    print(f"baseline={args.baseline} pickle_count={len(baseline_files)}")
    print("case\tkey\tfinite\tmax_abs_diff\tmean_abs_diff\trmse\tbaseline_rms")

    for case_dir in sorted(path for path in args.root.iterdir() if path.is_dir()):
        if case_dir.name in {args.baseline, "logs", "vanilla"}:
            continue
        files, values = load_case(case_dir)
        if not files:
            continue
        for key in ARRAY_KEYS:
            reference = baseline[key]
            value = values[key]
            if reference.shape != value.shape:
                print(f"{case_dir.name}\t{key}\tshape_mismatch:{value.shape}\t-\t-\t-\t-")
                continue
            difference = value - reference
            finite = bool(np.isfinite(value).all())
            print(
                f"{case_dir.name}\t{key}\t{finite}\t"
                f"{np.max(np.abs(difference)):.9g}\t"
                f"{np.mean(np.abs(difference)):.9g}\t"
                f"{np.sqrt(np.mean(np.square(difference))):.9g}\t"
                f"{np.sqrt(np.mean(np.square(reference))):.9g}"
            )


if __name__ == "__main__":
    main()
