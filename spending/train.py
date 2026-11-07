"""Run with python -m spending.train from the repository root."""
import argparse
import hashlib
import json
import platform
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from .model import DATA, ROOT, FEATURES, load_data, train

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--output", type=Path, default=ROOT / "reports")
    args = parser.parse_args()
    model, report, _, predictions = train(load_data(args.data))
    args.output.mkdir(parents=True, exist_ok=True)
    report["environment"] = {"python": platform.python_version(), "scikit_learn": sklearn.__version__, "pandas": pd.__version__}
    report["data_sha256"] = hashlib.sha256(args.data.read_bytes()).hexdigest()
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    pd.DataFrame({"feature": FEATURES, "coefficient": model.coef_}).to_csv(args.output / "coefficients.csv", index=False)
    predictions.to_csv(args.output / "holdout_predictions.csv", index_label="source_row")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    axes[0].scatter(predictions.actual, predictions.predicted, alpha=.7)
    lo, hi = predictions.actual.min(), predictions.actual.max()
    axes[0].plot([lo, hi], [lo, hi], "k--")
    axes[0].set(xlabel="Actual spending", ylabel="Predicted spending", title="Holdout predictions")
    axes[1].scatter(predictions.predicted, predictions.residual, alpha=.7)
    axes[1].axhline(0, color="black", linestyle="--")
    axes[1].set(xlabel="Predicted spending", ylabel="Actual - predicted", title="Residuals")
    axes[2].hist(predictions.residual, bins=20, edgecolor="white")
    axes[2].set(xlabel="Residual", ylabel="Count", title="Residual distribution")
    fig.savefig(args.output / "diagnostics.png", dpi=160)
    plt.close(fig)
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
