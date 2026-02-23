"""
Data cleaning module for the Telugu Multi-Turn Dialogue Research Pipeline.

Provides the DataCleaner class, which preprocesses and sanitizes Telugu
dialogue text. Handles missing values, removes unwanted artifacts such
as URLs and HTML tags, and normalizes whitespace while preserving the
full Telugu Unicode character set.
"""

import re
import logging
import pandas as pd
from config import Config


class DataCleaner:
    """
    Cleans and normalizes Telugu text data within a pandas DataFrame.

    Attributes:
        config: Pipeline configuration object.
        logger: Module-level logger.
    """

    def __init__(self, config: Config) -> None:
        """
        Initializes the DataCleaner with pipeline configuration.

        Args:
            config: Configuration object defining processing parameters.
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

    def clean_dataset(self, df: pd.DataFrame, text_column: str = "text") -> pd.DataFrame:
        """
        Applies cleaning transformations to a text column in the dataset.

        Args:
            df: DataFrame containing dialogue data.
            text_column: Name of the column containing dialogue text.

        Returns:
            A new DataFrame with cleaned text and null rows dropped.

        Raises:
            KeyError: If the specified text_column does not exist.
            TypeError: If the input is not a DataFrame.
        """
        if not isinstance(df, pd.DataFrame):
            self.logger.error("Input data is not a pandas DataFrame.")
            raise TypeError("Input df must be a pandas DataFrame.")

        if text_column not in df.columns:
            self.logger.error("Column '%s' not found in DataFrame.", text_column)
            raise KeyError(f"Column '{text_column}' is missing from the dataset.")

        self.logger.info("Starting data cleaning on column: %s", text_column)

        cleaned_df = df.copy()

        initial_count = len(cleaned_df)
        cleaned_df = cleaned_df.dropna(subset=[text_column])
        dropped_count = initial_count - len(cleaned_df)

        if dropped_count > 0:
            self.logger.info("Dropped %d rows with missing values.", dropped_count)

        cleaned_df[text_column] = cleaned_df[text_column].apply(self._clean_text)

        cleaned_df = cleaned_df[cleaned_df[text_column].str.strip() != ""]

        self.logger.info("Data cleaning complete. Final row count: %d", len(cleaned_df))
        return cleaned_df

    def _clean_text(self, text: str) -> str:
        """
        Performs cleaning on a single text string, preserving Telugu characters.

        Removes URLs, HTML tags, subtitle artifacts, and normalizes
        whitespace. All Telugu Unicode characters (U+0C00 to U+0C7F)
        are fully preserved.

        Args:
            text: The raw text string to clean.

        Returns:
            The sanitized and whitespace-normalized text string.
        """
        if not isinstance(text, str):
            self.logger.warning("Encountered non-string value during text cleaning.")
            return str(text)

        # Remove URLs
        text = re.sub(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$\-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
            '', text
        )

        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)

        # Remove common subtitle artifacts (e.g., timing codes, formatting tags)
        text = re.sub(r'\{\\[^}]*\}', '', text)

        # Remove leading/trailing subtitle credits and copyright markers
        text = re.sub(r'^[\u00a9\u00ae].*$', '', text, flags=re.MULTILINE)

        # Normalize multiple spaces to a single space
        text = re.sub(r'\s+', ' ', text)

        return text.strip()
