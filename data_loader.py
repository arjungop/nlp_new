"""
Data loading module for the Telugu Multi-Turn Dialogue Research Pipeline.

Parses IndicDialogue JSONL files, extracts Telugu dialogue lines from the
nested document structure, and applies strict language purity filtering to
ensure only lines written entirely in Telugu script are retained.
"""

import json
import re
import logging
import unicodedata
from typing import List, Dict, Any

import pandas as pd
from config import Config


class DataLoader:
    """
    Loads and filters Telugu dialogue data from IndicDialogue JSONL files.

    Each JSONL row represents one movie with the structure:
        {"ID": ..., "metadata": {...}, "dialogs": {"tel": [line1, line2, ...]}}

    The loader extracts individual dialogue lines from the "tel" key,
    filters out any line containing non-Telugu script characters, and
    returns a flat DataFrame suitable for downstream processing.

    Attributes:
        config: Pipeline configuration object.
        logger: Module-level logger.
    """

    _TELUGU_RANGE_START = 0x0C00
    _TELUGU_RANGE_END = 0x0C7F

    _ALLOWED_GENERAL_CATEGORIES = frozenset({
        "Zs",  # Space separator
        "Nd",  # Decimal digit
        "Po",  # Other punctuation
        "Ps",  # Open punctuation
        "Pe",  # Close punctuation
        "Pi",  # Initial quote
        "Pf",  # Final quote
        "Pd",  # Dash punctuation
        "Cc",  # Control characters (newline, tab)
        "Cf",  # Format characters (zero-width, etc.)
        "Mn",  # Non-spacing combining marks
        "Mc",  # Spacing combining marks
    })

    def __init__(self, config: Config) -> None:
        """
        Initializes the DataLoader with pipeline configuration.

        Args:
            config: Configuration object containing dataset file paths
                    and the Telugu purity threshold.
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

    def load_data(self, split: str = "train") -> pd.DataFrame:
        """
        Loads Telugu dialogue lines from a JSONL split file.

        Args:
            split: Which data split to load. Must be "train" or "test".

        Returns:
            A DataFrame with columns: movie_id, movie_name, year, text.
            Each row is a single dialogue line that passed purity filtering.

        Raises:
            ValueError: If split is not "train" or "test".
            FileNotFoundError: If the JSONL file does not exist.
        """
        if split not in ("train", "test"):
            raise ValueError(f"Split must be 'train' or 'test'. Received: '{split}'.")

        file_path = self.config.train_data_path if split == "train" else self.config.test_data_path
        self.logger.info("Loading %s split from: %s", split, file_path)

        records = self._parse_jsonl(file_path)
        raw_lines = self._extract_dialogue_lines(records)
        pure_lines = self._filter_telugu_lines(raw_lines)

        df = pd.DataFrame(pure_lines, columns=["movie_id", "movie_name", "year", "text"])

        self.logger.info(
            "Loaded %d pure Telugu lines from %d movies (%s split).",
            len(df), df["movie_id"].nunique(), split
        )

        return df

    def _parse_jsonl(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Reads a JSONL file and returns a list of parsed dictionaries.

        Args:
            file_path: Absolute path to the JSONL file.

        Returns:
            A list of dictionaries, one per line in the file.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        records: List[Dict[str, Any]] = []

        with open(file_path, "r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    records.append(json.loads(stripped))
                except json.JSONDecodeError as e:
                    self.logger.warning(
                        "Skipping malformed JSON at line %d in %s: %s",
                        line_number, file_path, str(e)
                    )

        self.logger.info("Parsed %d records from %s.", len(records), file_path)
        return records

    def _extract_dialogue_lines(
        self, records: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Extracts individual dialogue lines from parsed JSONL records.

        Args:
            records: List of movie-level dictionaries with "dialogs" key.

        Returns:
            A list of dictionaries, each containing movie_id, movie_name,
            year, and a single dialogue line in the "text" field.
        """
        all_lines: List[Dict[str, Any]] = []

        for record in records:
            movie_id = str(record.get("ID", ""))
            metadata = record.get("metadata", {})
            movie_name = str(metadata.get("MovieName", ""))
            year = metadata.get("Year")

            dialogs = record.get("dialogs", {})
            tel_lines = dialogs.get("tel", [])

            if not isinstance(tel_lines, list):
                self.logger.warning(
                    "Movie %s has non-list 'tel' dialogs. Skipping.", movie_id
                )
                continue

            for line in tel_lines:
                if not isinstance(line, str):
                    continue
                cleaned = line.strip()
                if not cleaned:
                    continue
                all_lines.append({
                    "movie_id": movie_id,
                    "movie_name": movie_name,
                    "year": year,
                    "text": cleaned,
                })

        self.logger.info("Extracted %d raw dialogue lines.", len(all_lines))
        return all_lines

    def _filter_telugu_lines(
        self, lines: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Retains only lines where every script character is Telugu.

        A line is considered pure Telugu if all characters that carry script
        identity (letters, syllabic marks) fall within Unicode block U+0C00
        to U+0C7F. Whitespace, digits, and punctuation are script-neutral
        and do not affect the purity calculation.

        Lines that contain zero script characters (only digits, punctuation,
        or whitespace) are discarded.

        Args:
            lines: List of dialogue line dictionaries with a "text" field.

        Returns:
            The filtered list containing only pure Telugu entries.
        """
        pure: List[Dict[str, Any]] = []
        rejected = 0

        for entry in lines:
            text = entry["text"]
            if self._is_pure_telugu(text):
                pure.append(entry)
            else:
                rejected += 1

        self.logger.info(
            "Telugu purity filter: accepted %d, rejected %d.",
            len(pure), rejected
        )
        return pure

    def _is_pure_telugu(self, text: str) -> bool:
        """
        Determines whether a text string contains exclusively Telugu script.

        Every character that is a letter or syllabic mark must fall within
        the Telugu Unicode range. Characters that are script-neutral
        (whitespace, digits, punctuation, control characters) are ignored
        in the purity check but the text must contain at least one actual
        Telugu character to be accepted.

        Args:
            text: The string to evaluate.

        Returns:
            True if the text is pure Telugu, False otherwise.
        """
        telugu_char_count = 0

        for char in text:
            code_point = ord(char)
            category = unicodedata.category(char)

            if self._TELUGU_RANGE_START <= code_point <= self._TELUGU_RANGE_END:
                telugu_char_count += 1
                continue

            if category in self._ALLOWED_GENERAL_CATEGORIES:
                continue

            # Any character that is a letter, symbol, or mark outside
            # the Telugu block means this line is contaminated.
            return False

        return telugu_char_count > 0