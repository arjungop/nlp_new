"""
Dialogue segmentation module for the Telugu Multi-Turn Dialogue Research Pipeline.

Transforms a flat DataFrame of ordered dialogue lines into context-response
pairs using a sliding window approach. Context windows are constructed
per-movie to maintain conversational coherence within a single film's
dialogue sequence.
"""

import logging
from typing import List, Dict

import pandas as pd
from config import Config


class DialogueSegmenter:
    """
    Segments sequential dialogue lines into multi-turn context-response pairs.

    For each movie's dialogue sequence, a sliding window of size
    config.context_window_size is advanced one line at a time. The window
    contents form the context, and the immediately following line is the
    expected response.

    Attributes:
        config: Pipeline configuration object.
        logger: Module-level logger.
    """

    def __init__(self, config: Config) -> None:
        """
        Initializes the DialogueSegmenter with pipeline configuration.

        Args:
            config: Configuration object containing context_window_size.
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

    def segment_dialogues(
        self,
        df: pd.DataFrame,
        text_column: str = "text",
    ) -> List[Dict[str, str]]:
        """
        Produces context-response pairs from an ordered dialogue DataFrame.

        Dialogues are grouped by movie_id to prevent cross-movie context
        contamination. Within each movie, a sliding window generates one
        pair per valid position.

        Args:
            df: DataFrame with at least a "movie_id" column and the
                specified text column.
            text_column: Name of the column containing dialogue text.

        Returns:
            A list of dictionaries with keys "context" and "response".
            The "context" value is a newline-joined string of window_size
            consecutive lines, and "response" is the next line.

        Raises:
            KeyError: If required columns are missing from the DataFrame.
        """
        required_columns = {"movie_id", text_column}
        missing = required_columns - set(df.columns)
        if missing:
            raise KeyError(f"Missing required columns: {missing}")

        window_size = self.config.context_window_size
        pairs: List[Dict[str, str]] = []

        grouped = df.groupby("movie_id", sort=False)

        for movie_id, group_df in grouped:
            lines = group_df[text_column].tolist()

            if len(lines) <= window_size:
                self.logger.debug(
                    "Movie %s has %d lines, fewer than window size %d. Skipping.",
                    movie_id, len(lines), window_size
                )
                continue

            for i in range(len(lines) - window_size):
                context_lines = lines[i : i + window_size]
                response_line = lines[i + window_size]

                context = "\n".join(context_lines)

                pairs.append({
                    "context": context,
                    "response": response_line,
                })

        self.logger.info(
            "Generated %d context-response pairs from %d movies "
            "(window size: %d).",
            len(pairs), grouped.ngroups, window_size
        )

        return pairs