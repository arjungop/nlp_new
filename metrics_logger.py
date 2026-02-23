"""
Metrics logging module for the Telugu Multi-Turn Dialogue Research Pipeline.

This module provides the MetricsLogger class, which is responsible for persisting
the computed evaluation metrics and generated texts to structured formats (CSV and JSON)
for archival, further analysis, and visualization.
"""

import os
import csv
import json
import logging
from typing import List, Dict, Any
from config import Config


class MetricsLogger:
    """
    Handles the structured logging of evaluation results to disk.

    Attributes:
        config: An instance of the Config dataclass containing output directory paths.
        logger: A standard Python logger for tracking file I/O operations.
    """

    def __init__(self, config: Config) -> None:
        """
        Initializes the MetricsLogger and ensures the output directories exist.

        Args:
            config: Configuration object defining output directory paths.

        Raises:
            OSError: If the output directory cannot be created due to permission issues.
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        try:
            os.makedirs(self.config.output_dir, exist_ok=True)
            self.logger.info("Output directory ensured at: %s", self.config.output_dir)
        except OSError as e:
            self.logger.error("Failed to create output directory: %s", str(e))
            raise OSError(f"Cannot create directory at {self.config.output_dir}.") from e

    def log_results(self, results: List[Dict[str, Any]]) -> None:
        """
        Writes the evaluation results to both CSV and JSON formats.

        Args:
            results: A list of dictionaries containing dialogue contexts, 
                     generated responses, and computed similarity metrics.

        Raises:
            TypeError: If the results input is not a list.
            ValueError: If the results list is empty.
            IOError: If writing to the CSV or JSON file fails.
        """
        if not isinstance(results, list):
            self.logger.error("Results must be provided as a list.")
            raise TypeError("Expected a list of evaluation dictionaries.")

        if not results:
            self.logger.error("Cannot log empty results list.")
            raise ValueError("The results list is empty.")

        self.logger.info("Logging evaluation results for %d items.", len(results))

        csv_path = os.path.join(self.config.output_dir, "evaluation_results.csv")
        json_path = os.path.join(self.config.output_dir, "evaluation_results.json")

        self._write_to_csv(results, csv_path)
        self._write_to_json(results, json_path)

        self.logger.info("Successfully logged all metrics to %s.", self.config.output_dir)

    def _write_to_csv(self, results: List[Dict[str, Any]], filepath: str) -> None:
        """
        Writes the results list to a structured CSV file.

        Args:
            results: The data to write.
            filepath: The target file path.

        Raises:
            IOError: If a file operation fails.
        """
        try:
            # Extract headers from the first dictionary
            headers = list(results[0].keys())
            
            with open(filepath, mode="w", newline="", encoding="utf-8") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=headers)
                writer.writeheader()
                writer.writerows(results)
                
            self.logger.debug("Successfully saved CSV to %s", filepath)
        except IOError as e:
            self.logger.error("Failed to write CSV file at %s: %s", filepath, str(e))
            raise IOError("Could not write evaluation results to CSV.") from e
        except Exception as e:
            self.logger.error("Unexpected error writing CSV: %s", str(e))
            raise IOError("Unexpected failure while writing CSV.") from e

    def _write_to_json(self, results: List[Dict[str, Any]], filepath: str) -> None:
        """
        Writes the results list to a formatted JSON file.

        Args:
            results: The data to write.
            filepath: The target file path.

        Raises:
            IOError: If a file operation fails.
            TypeError: If the data contains unserializable objects.
        """
        try:
            with open(filepath, mode="w", encoding="utf-8") as json_file:
                json.dump(results, json_file, indent=4, ensure_ascii=False)
                
            self.logger.debug("Successfully saved JSON to %s", filepath)
        except IOError as e:
            self.logger.error("Failed to write JSON file at %s: %s", filepath, str(e))
            raise IOError("Could not write evaluation results to JSON.") from e
        except TypeError as e:
            self.logger.error("JSON serialization error: %s", str(e))
            raise TypeError("Results contain data that cannot be serialized to JSON.") from e