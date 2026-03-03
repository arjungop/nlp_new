"""
Standalone statistical analysis script for Telugu Dialogue Pipeline results.

Run locally after downloading evaluation_results.csv from HuggingFace:
    huggingface-cli download arjg/IndicNLP outputs/evaluation_results.csv \
        --repo-type dataset --local-dir outputs/

Usage:
    python3 statistical_analysis.py
    python3 statistical_analysis.py --json   # use evaluation_checkpoint.json instead
"""

import argparse
import json
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# ──────────────────────────────────────────────
MODELS    = ["t5_raw", "t5_cot", "sarvam_raw"]
MODEL_LABELS = {
    "t5_raw":     "Gemma RAW",
    "t5_cot":     "Gemma CoT",
    "sarvam_raw": "Sarvam RAW",
}
METRICS   = ["cosine", "jaccard", "dice", "bert_f1"]
METRIC_LABELS = {
    "cosine":  "Cosine Similarity",
    "jaccard": "Jaccard Similarity",
    "dice":    "Dice Similarity",
    "bert_f1": "BERTScore F1",
}
OUTPUT_DIR = "outputs"
# ──────────────────────────────────────────────


def load_data(use_json: bool = False) -> pd.DataFrame:
    if use_json:
        path = os.path.join(OUTPUT_DIR, "evaluation_checkpoint.json")
        print(f"Loading {path}...")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        df = pd.DataFrame(data)
    else:
        path = os.path.join(OUTPUT_DIR, "evaluation_results.csv")
        print(f"Loading {path}...")
        df = pd.read_csv(path)

    print(f"Loaded {len(df):,} rows, {len(df.columns)} columns.\n")
    return df


def descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Mean, std, median, min, max, IQR for each model×metric."""
    rows = []
    for model in MODELS:
        for metric in METRICS:
            col = f"{model}_{metric}"
            if col not in df.columns:
                continue
            vals = df[col].dropna()
            rows.append({
                "Model":    MODEL_LABELS[model],
                "Metric":   METRIC_LABELS[metric],
                "N":        len(vals),
                "Mean":     vals.mean(),
                "Std":      vals.std(),
                "Median":   vals.median(),
                "Min":      vals.min(),
                "Max":      vals.max(),
                "IQR":      vals.quantile(0.75) - vals.quantile(0.25),
            })
    return pd.DataFrame(rows)


def pairwise_ttests(df: pd.DataFrame) -> pd.DataFrame:
    """Paired t-test for every model pair on every metric."""
    from itertools import combinations
    rows = []
    for metric in METRICS:
        for m1, m2 in combinations(MODELS, 2):
            c1, c2 = f"{m1}_{metric}", f"{m2}_{metric}"
            if c1 not in df.columns or c2 not in df.columns:
                continue
            a = df[c1].dropna()
            b = df[c2].dropna()
            n = min(len(a), len(b))
            t_stat, p_val = stats.ttest_rel(a[:n], b[:n])
            rows.append({
                "Metric":   METRIC_LABELS[metric],
                "Model A":  MODEL_LABELS[m1],
                "Model B":  MODEL_LABELS[m2],
                "t-stat":   round(t_stat, 4),
                "p-value":  round(p_val, 6),
                "Sig (p<0.05)": "✓" if p_val < 0.05 else "✗",
                "Winner":   MODEL_LABELS[m1] if a[:n].mean() > b[:n].mean() else MODEL_LABELS[m2],
            })
    return pd.DataFrame(rows)


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Effect size: Cohen's d."""
    pooled_std = np.sqrt((np.std(a, ddof=1)**2 + np.std(b, ddof=1)**2) / 2)
    return (np.mean(a) - np.mean(b)) / pooled_std if pooled_std > 0 else 0.0


def effect_sizes(df: pd.DataFrame) -> pd.DataFrame:
    """Cohen's d effect size for each model pair × metric."""
    from itertools import combinations
    rows = []
    for metric in METRICS:
        for m1, m2 in combinations(MODELS, 2):
            c1, c2 = f"{m1}_{metric}", f"{m2}_{metric}"
            if c1 not in df.columns or c2 not in df.columns:
                continue
            a = df[c1].dropna().values
            b = df[c2].dropna().values
            n = min(len(a), len(b))
            d = cohens_d(a[:n], b[:n])
            magnitude = (
                "negligible" if abs(d) < 0.2 else
                "small"      if abs(d) < 0.5 else
                "medium"     if abs(d) < 0.8 else
                "large"
            )
            rows.append({
                "Metric":    METRIC_LABELS[metric],
                "Model A":   MODEL_LABELS[m1],
                "Model B":   MODEL_LABELS[m2],
                "Cohen's d": round(d, 4),
                "Magnitude": magnitude,
            })
    return pd.DataFrame(rows)


def wilcoxon_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Wilcoxon signed-rank test (non-parametric alternative to paired t-test)."""
    from itertools import combinations
    rows = []
    for metric in METRICS:
        for m1, m2 in combinations(MODELS, 2):
            c1, c2 = f"{m1}_{metric}", f"{m2}_{metric}"
            if c1 not in df.columns or c2 not in df.columns:
                continue
            a = df[c1].dropna()
            b = df[c2].dropna()
            n = min(len(a), len(b))
            try:
                stat, p_val = stats.wilcoxon(a[:n], b[:n])
                rows.append({
                    "Metric":        METRIC_LABELS[metric],
                    "Model A":       MODEL_LABELS[m1],
                    "Model B":       MODEL_LABELS[m2],
                    "W-stat":        round(stat, 2),
                    "p-value":       round(p_val, 6),
                    "Sig (p<0.05)":  "✓" if p_val < 0.05 else "✗",
                })
            except Exception as e:
                print(f"  Wilcoxon failed for {m1} vs {m2} on {metric}: {e}")
    return pd.DataFrame(rows)


def win_rate(df: pd.DataFrame) -> pd.DataFrame:
    """Per-row win rate: how often does each model score highest?"""
    rows = []
    for metric in METRICS:
        cols = {m: f"{m}_{metric}" for m in MODELS if f"{m}_{metric}" in df.columns}
        if len(cols) < 2:
            continue
        sub = df[[c for c in cols.values()]].dropna()
        winner = sub.idxmax(axis=1)
        for model, col in cols.items():
            win_count = (winner == col).sum()
            rows.append({
                "Metric":    METRIC_LABELS[metric],
                "Model":     MODEL_LABELS[model],
                "Wins":      int(win_count),
                "Win Rate":  f"{100 * win_count / len(sub):.1f}%",
            })
    return pd.DataFrame(rows)


def plot_distributions(df: pd.DataFrame, out_dir: str) -> None:
    """Violin + box plots for each metric."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    palette = ["#4C72B0", "#DD8452", "#55A868"]

    for i, metric in enumerate(METRICS):
        ax = axes[i]
        plot_data = []
        for model in MODELS:
            col = f"{model}_{metric}"
            if col in df.columns:
                vals = df[col].dropna().values
                plot_data.extend([(MODEL_LABELS[model], v) for v in vals])

        plot_df = pd.DataFrame(plot_data, columns=["Model", "Score"])
        sns.violinplot(data=plot_df, x="Model", y="Score",
                       palette=palette, inner="box", ax=ax, cut=0)
        ax.set_title(METRIC_LABELS[metric], fontsize=13, fontweight="bold")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=15)

    plt.suptitle("Score Distributions by Model & Prompting Strategy", fontsize=15, y=1.01)
    plt.tight_layout()
    path = os.path.join(out_dir, "violin_distributions.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_correlation_matrix(df: pd.DataFrame, out_dir: str) -> None:
    """Correlation between all metric columns."""
    cols = [f"{m}_{met}" for m in MODELS for met in METRICS if f"{m}_{met}" in df.columns]
    corr = df[cols].corr()
    labels = [f"{MODEL_LABELS[m.rsplit('_',1)[0]] if m.rsplit('_',1)[0] in MODEL_LABELS else m}\n{METRIC_LABELS.get(m.rsplit('_',1)[-1], m.rsplit('_',1)[-1])}"
              for m in cols]

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                xticklabels=labels, yticklabels=labels,
                center=0, linewidths=0.5, ax=ax)
    ax.set_title("Inter-Metric Correlation Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir, "correlation_matrix.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def print_section(title: str) -> None:
    width = 70
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true", help="Use evaluation_checkpoint.json instead of CSV")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = load_data(use_json=args.json)

    # ── 1. Descriptive Stats ──────────────────────────────────────────────
    print_section("1. Descriptive Statistics")
    desc = descriptive_stats(df)
    print(desc.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    desc.to_csv(os.path.join(OUTPUT_DIR, "stats_descriptive.csv"), index=False)

    # ── 2. Paired t-Tests ─────────────────────────────────────────────────
    print_section("2. Paired t-Tests (parametric)")
    ttest = pairwise_ttests(df)
    print(ttest.to_string(index=False))
    ttest.to_csv(os.path.join(OUTPUT_DIR, "stats_ttests.csv"), index=False)

    # ── 3. Wilcoxon Signed-Rank Tests ─────────────────────────────────────
    print_section("3. Wilcoxon Signed-Rank Tests (non-parametric)")
    wilcox = wilcoxon_tests(df)
    print(wilcox.to_string(index=False))
    wilcox.to_csv(os.path.join(OUTPUT_DIR, "stats_wilcoxon.csv"), index=False)

    # ── 4. Effect Sizes ───────────────────────────────────────────────────
    print_section("4. Effect Sizes (Cohen's d)")
    eff = effect_sizes(df)
    print(eff.to_string(index=False))
    eff.to_csv(os.path.join(OUTPUT_DIR, "stats_effect_sizes.csv"), index=False)

    # ── 5. Win Rates ──────────────────────────────────────────────────────
    print_section("5. Per-Row Win Rates")
    wins = win_rate(df)
    print(wins.to_string(index=False))
    wins.to_csv(os.path.join(OUTPUT_DIR, "stats_win_rates.csv"), index=False)

    # ── 6. Plots ──────────────────────────────────────────────────────────
    print_section("6. Generating Plots")
    plot_distributions(df, OUTPUT_DIR)
    plot_correlation_matrix(df, OUTPUT_DIR)

    print_section("DONE")
    print(f"  All CSVs and plots saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
