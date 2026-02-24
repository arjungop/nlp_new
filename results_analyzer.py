"""
Results analysis module for the Telugu Multi-Turn Dialogue Research Pipeline.

Provides the ResultsAnalyzer class, which aggregates per-sample evaluation
metrics into a structured summary matrix and renders formatted console output
for rapid performance comparison across models and prompting strategies.
"""

import logging
from typing import List, Dict, Any

from config import Config


_MODEL_PROMPT_COMBOS = ("t5_raw", "t5_cot", "sarvam_raw")
_METRIC_NAMES = ("cosine", "jaccard", "dice", "bert_f1")


class ResultsAnalyzer:
    """
    Aggregates evaluation metrics and produces summary statistics.

    Attributes:
        config: Pipeline configuration object.
        logger: Module-level logger.
    """

    def __init__(self, config: Config) -> None:
        """
        Initializes the ResultsAnalyzer.

        Args:
            config: Pipeline configuration object.
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

    def analyze_metrics(
        self, evaluation_results: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, float]]:
        """
        Computes mean scores for each model-prompt and metric combination.

        Args:
            evaluation_results: List of per-sample evaluation dictionaries
                produced by the Evaluator.

        Returns:
            A nested dictionary mapping each model-prompt key to a dictionary
            of metric names and their mean values. Structure:
                { "t5_raw": {"cosine": 0.82, "jaccard": 0.31, "dice": 0.47, "bert_f1": 0.75}, ... }

        Raises:
            ValueError: If the results list is empty.
        """
        if not evaluation_results:
            raise ValueError("Cannot analyze an empty results list.")

        summary: Dict[str, Dict[str, float]] = {}

        for combo in _MODEL_PROMPT_COMBOS:
            combo_metrics: Dict[str, float] = {}

            for metric in _METRIC_NAMES:
                key = f"{combo}_{metric}"
                values = [
                    row[key]
                    for row in evaluation_results
                    if key in row and isinstance(row[key], (int, float))
                ]

                if values:
                    combo_metrics[metric] = sum(values) / len(values)
                else:
                    combo_metrics[metric] = 0.0

            summary[combo] = combo_metrics

        self.logger.info("Metric analysis complete for %d model-prompt combinations.",
                         len(summary))
        return summary

    def print_summary(self, summary_matrix: Dict[str, Dict[str, float]]) -> None:
        """
        Prints a formatted table of aggregated metrics to the console.

        Args:
            summary_matrix: The dictionary returned by analyze_metrics.
        """
        header = f"{'Model+Prompt':<20}"
        for metric in _METRIC_NAMES:
            header += f"{metric.capitalize():>12}"
        table_width = 20 + 12 * len(_METRIC_NAMES)
        print("\n" + "=" * table_width)
        print("  Performance Summary")
        print("=" * table_width)
        print(header)
        print("-" * table_width)

        for combo in _MODEL_PROMPT_COMBOS:
            scores = summary_matrix.get(combo, {})
            row = f"{combo:<20}"
            for metric in _METRIC_NAMES:
                value = scores.get(metric, 0.0)
                row += f"{value:>12.4f}"
            print(row)

        print("=" * table_width + "\n")