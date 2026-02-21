from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def save_example_plots_from_normalized_csvs(
    csv_files: list[Path],
    out_dir: Path,
    n_plots: int = 12,
    resample_rule: str | None = "D",  # "D", "H", or None for raw
    seed: int = 0,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    good = []
    for p in csv_files:
        try:
            cols = pd.read_csv(p, nrows=1).columns
            if {"utc", "min", "max"}.issubset(set(cols)):
                good.append(p)
        except Exception:
            pass

    if not good:
        print("[PLOTS] No CSVs with utc/min/max found to plot.")
        return

    picks = rng.choice(good, size=min(n_plots, len(good)), replace=False)

    for p in picks:
        try:
            df = pd.read_csv(p)
            df["utc"] = pd.to_datetime(df["utc"], errors="coerce")
            df = df.dropna(subset=["utc"]).sort_values("utc")

            if resample_rule is not None:
                df = df.set_index("utc")[["min", "max"]].resample(resample_rule).mean()

            plt.figure(figsize=(12, 5))
            if resample_rule is None:
                plt.plot(df["utc"], df["min"], label="min (normalized)")
                plt.plot(df["utc"], df["max"], label="max (normalized)")
                plt.xlabel("Time (UTC)")
            else:
                plt.plot(df.index, df["min"], label=f"{resample_rule} mean(min) normalized")
                plt.plot(df.index, df["max"], label=f"{resample_rule} mean(max) normalized")
                plt.xlabel("Time (UTC)")

            plt.ylabel("Normalized power")
            plt.title(p.stem)
            plt.legend()
            plt.tight_layout()

            out_png = out_dir / f"{p.stem}.png"
            plt.savefig(out_png, dpi=150)
            plt.close()
        except Exception as e:
            print(f"[PLOTS][ERR] {p}: {e}")

    print(f"[PLOTS] Saved {min(n_plots, len(good))} plots to: {out_dir.resolve()}")
