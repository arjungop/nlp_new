"""
Visualization module for the Telugu Multi-Turn Dialogue Research Pipeline.

Provides the Visualizer class, which generates publication-quality heatmaps
and grouped bar charts from the aggregated evaluation summary matrix.
All plots are saved to the configured output directory.
"""

import os
import logging
from typing import Dict, List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from config import Config


_MODEL_PROMPT_COMBOS: List[str] = ["t5_raw", "t5_cot", "sarvam_raw"]
_METRIC_NAMES: List[str] = ["cosine", "jaccard", "dice"]

_DISPLAY_LABELS: Dict[str, str] = {
    "t5_raw": "IndicT5 Raw",
    "t5_cot": "IndicT5 COT",
    "sarvam_raw": "Sarvam Raw",
}


class Visualizer:
    """
    Generates and saves evaluation visualizations.

    Attributes:
        config: Pipeline configuration object.
        logger: Module-level logger.
    """

    def __init__(self, config: Config) -> None:
        """
        Initializes the Visualizer, ensuring the output directory exists.

        Args:
            config: Pipeline configuration object.
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        os.makedirs(self.config.output_dir, exist_ok=True)

    def plot_heatmap(self, summary_matrix: Dict[str, Dict[str, float]]) -> None:
        """
        Renders a heatmap of model-prompt scores across all metrics.

        Args:
            summary_matrix: Nested dictionary from ResultsAnalyzer.analyze_metrics.
        """
        data = []
        row_labels = []

        for combo in _MODEL_PROMPT_COMBOS:
            scores = summary_matrix.get(combo, {})
            row_labels.append(_DISPLAY_LABELS.get(combo, combo))
            data.append([scores.get(m, 0.0) for m in _METRIC_NAMES])

        matrix = np.array(data)
        col_labels = [m.capitalize() for m in _METRIC_NAMES]

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.heatmap(
            matrix,
            annot=True,
            fmt=".4f",
            cmap="YlOrRd",
            xticklabels=col_labels,
            yticklabels=row_labels,
            linewidths=0.5,
            ax=ax,
        )
        ax.set_title("Telugu Dialogue Response Evaluation Heatmap")
        ax.set_xlabel("Metric")
        ax.set_ylabel("Model + Prompt Strategy")

        fig.tight_layout()
        output_path = os.path.join(self.config.output_dir, "heatmap.png")
        fig.savefig(output_path, dpi=150)
        plt.close(fig)

        self.logger.info("Heatmap saved to %s", output_path)

    def plot_bar_charts(self, summary_matrix: Dict[str, Dict[str, float]]) -> None:
        """
        Renders a grouped bar chart comparing models across metrics.

        Args:
            summary_matrix: Nested dictionary from ResultsAnalyzer.analyze_metrics.
        """
        num_combos = len(_MODEL_PROMPT_COMBOS)
        num_metrics = len(_METRIC_NAMES)
        x = np.arange(num_metrics)
        bar_width = 0.8 / num_combos

        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63"]

        for i, combo in enumerate(_MODEL_PROMPT_COMBOS):
            scores = summary_matrix.get(combo, {})
            values = [scores.get(m, 0.0) for m in _METRIC_NAMES]
            offset = (i - num_combos / 2 + 0.5) * bar_width
            label = _DISPLAY_LABELS.get(combo, combo)
            ax.bar(x + offset, values, bar_width, label=label, color=colors[i])

        ax.set_xlabel("Metric")
        ax.set_ylabel("Mean Score")
        ax.set_title("Telugu Dialogue Response Evaluation Comparison")
        ax.set_xticks(x)
        ax.set_xticklabels([m.capitalize() for m in _METRIC_NAMES])
        ax.legend()
        ax.set_ylim(0, 1.0)

        fig.tight_layout()
        output_path = os.path.join(self.config.output_dir, "bar_chart.png")
        fig.savefig(output_path, dpi=150)
        plt.close(fig)

        self.logger.info("Bar chart saved to %s", output_path)