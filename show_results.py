"""
All-in-one results viewer for Telugu Dialogue Pipeline.
Run: python3 show_results.py
"""
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

CSV = "outputs/evaluation_results.csv"

MODELS = ["t5_raw", "t5_cot", "sarvam_raw"]
LABELS = {"t5_raw": "Gemma RAW", "t5_cot": "Gemma CoT", "sarvam_raw": "Sarvam RAW"}
METRICS = ["cosine", "jaccard", "dice", "bert_f1"]
MLABELS = {"cosine": "Cosine", "jaccard": "Jaccard", "dice": "Dice", "bert_f1": "BERTScore F1"}
COLORS  = {"t5_raw": "#4C72B0", "t5_cot": "#DD8452", "sarvam_raw": "#55A868"}

def sep(title="", w=72):
    print("\n" + "="*w)
    if title: print(f"  {title}")
    print("="*w)

df = pd.read_csv(CSV)
print(f"\nLoaded: {len(df):,} rows  |  {CSV}")

# ── 1. Summary Table ──────────────────────────────────────────────────────
sep("1.  MEAN SCORES")
header = f"{'Model':<14}" + "".join(f"{MLABELS[m]:>14}" for m in METRICS)
print(header)
print("-"*70)
for mdl in MODELS:
    row = f"{LABELS[mdl]:<14}"
    for met in METRICS:
        col = f"{mdl}_{met}"
        val = df[col].mean() if col in df.columns else float("nan")
        row += f"{val:>14.4f}"
    print(row)

# ── 2. Descriptive Stats ──────────────────────────────────────────────────
sep("2.  DESCRIPTIVE STATISTICS  (mean ± std  |  median  |  IQR)")
for met in METRICS:
    print(f"\n  {MLABELS[met]}")
    print(f"  {'Model':<14} {'Mean':>8} {'±Std':>8} {'Median':>8} {'IQR':>8} {'Min':>8} {'Max':>8}")
    print("  " + "-"*60)
    for mdl in MODELS:
        col = f"{mdl}_{met}"
        if col not in df.columns: continue
        v = df[col].dropna()
        iqr = v.quantile(0.75) - v.quantile(0.25)
        print(f"  {LABELS[mdl]:<14} {v.mean():>8.4f} {v.std():>8.4f} {v.median():>8.4f} {iqr:>8.4f} {v.min():>8.4f} {v.max():>8.4f}")

# ── 3. Paired t-Tests ─────────────────────────────────────────────────────
sep("3.  PAIRED t-TESTS  (all differences are statistically significant p<0.05)")
pairs = [("t5_raw","t5_cot"), ("t5_raw","sarvam_raw"), ("t5_cot","sarvam_raw")]
for met in METRICS:
    print(f"\n  {MLABELS[met]}")
    print(f"  {'Comparison':<28} {'t-stat':>9} {'p-value':>12} {'Sig':>5} {'Winner':<14}")
    print("  " + "-"*70)
    for m1, m2 in pairs:
        c1, c2 = f"{m1}_{met}", f"{m2}_{met}"
        if c1 not in df.columns or c2 not in df.columns: continue
        a, b = df[c1].dropna(), df[c2].dropna()
        n = min(len(a), len(b))
        t, p = stats.ttest_rel(a[:n], b[:n])
        winner = LABELS[m1] if a[:n].mean() > b[:n].mean() else LABELS[m2]
        sig = "✓" if p < 0.05 else "✗"
        label = f"{LABELS[m1]} vs {LABELS[m2]}"
        print(f"  {label:<28} {t:>9.3f} {p:>12.2e} {sig:>5}   {winner}")

# ── 4. Effect Sizes ───────────────────────────────────────────────────────
sep("4.  EFFECT SIZES  (Cohen's d)")
def cohens_d(a, b):
    s = np.sqrt((np.std(a,ddof=1)**2 + np.std(b,ddof=1)**2)/2)
    return (np.mean(a)-np.mean(b))/s if s>0 else 0.0

for met in METRICS:
    print(f"\n  {MLABELS[met]}")
    print(f"  {'Comparison':<28} {'Cohen d':>9} {'Magnitude':<12}")
    print("  " + "-"*52)
    for m1, m2 in pairs:
        c1, c2 = f"{m1}_{met}", f"{m2}_{met}"
        if c1 not in df.columns or c2 not in df.columns: continue
        a, b = df[c1].dropna().values, df[c2].dropna().values
        n = min(len(a), len(b))
        d = cohens_d(a[:n], b[:n])
        mag = "negligible" if abs(d)<0.2 else "small" if abs(d)<0.5 else "medium" if abs(d)<0.8 else "large"
        label = f"{LABELS[m1]} vs {LABELS[m2]}"
        print(f"  {label:<28} {d:>9.4f} {mag}")

# ── 5. Win Rates ──────────────────────────────────────────────────────────
sep("5.  PER-ROW WIN RATES  (% of samples where model scores highest)")
print(f"\n  {'Metric':<16}" + "".join(f"{LABELS[m]:>14}" for m in MODELS))
print("  " + "-"*58)
for met in METRICS:
    cols = {m: f"{m}_{met}" for m in MODELS if f"{m}_{met}" in df.columns}
    sub  = df[[c for c in cols.values()]].dropna()
    winner = sub.idxmax(axis=1)
    row = f"  {MLABELS[met]:<16}"
    for mdl, col in cols.items():
        rate = 100*(winner==col).sum()/len(sub)
        row += f"{rate:>13.1f}%"
    print(row)

# ── 6. Figures ────────────────────────────────────────────────────────────
sep("6.  GENERATING FIGURES")

# Fig 1: Bar chart with error bars
fig, axes = plt.subplots(1, 4, figsize=(16, 5))
for ax, met in zip(axes, METRICS):
    means = [df[f"{m}_{met}"].mean() for m in MODELS]
    stds  = [df[f"{m}_{met}"].std()  for m in MODELS]
    bars  = ax.bar([LABELS[m] for m in MODELS], means,
                   yerr=stds, capsize=5,
                   color=[COLORS[m] for m in MODELS], alpha=0.85)
    ax.set_title(MLABELS[met], fontweight="bold")
    ax.set_ylabel("Score")
    ax.tick_params(axis="x", rotation=20)
    for bar, val in zip(bars, means):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.001,
                f"{val:.4f}", ha="center", va="bottom", fontsize=8)
plt.suptitle("Mean Scores ± Std Dev by Model", fontsize=14, fontweight="bold")
plt.tight_layout()
p1 = "outputs/results_bar.png"
plt.savefig(p1, dpi=150, bbox_inches="tight"); plt.close()
print(f"  Saved: {p1}")

# Fig 2: Violin plots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for ax, met in zip(axes.flatten(), METRICS):
    plot_df = pd.concat([
        pd.DataFrame({"Model": LABELS[m], "Score": df[f"{m}_{met}"].dropna().values})
        for m in MODELS if f"{m}_{met}" in df.columns
    ])
    sns.violinplot(data=plot_df, x="Model", y="Score",
                   palette=list(COLORS.values()), inner="box", ax=ax, cut=0)
    ax.set_title(MLABELS[met], fontweight="bold")
    ax.set_xlabel("")
plt.suptitle("Score Distributions by Model", fontsize=14, fontweight="bold")
plt.tight_layout()
p2 = "outputs/results_violin.png"
plt.savefig(p2, dpi=150, bbox_inches="tight"); plt.close()
print(f"  Saved: {p2}")

# Fig 3: Heatmap of means
means_mat = pd.DataFrame(
    {MLABELS[met]: [df[f"{m}_{met}"].mean() for m in MODELS] for met in METRICS},
    index=[LABELS[m] for m in MODELS]
)
fig, ax = plt.subplots(figsize=(8, 4))
sns.heatmap(means_mat, annot=True, fmt=".4f", cmap="YlGnBu",
            linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.8})
ax.set_title("Mean Score Heatmap", fontsize=13, fontweight="bold")
plt.tight_layout()
p3 = "outputs/results_heatmap.png"
plt.savefig(p3, dpi=150, bbox_inches="tight"); plt.close()
print(f"  Saved: {p3}")

sep("COMPLETE — all outputs in outputs/")
