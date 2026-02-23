"""
Triplet building module for the Telugu Multi-Turn Dialogue Research Pipeline.

This module provides the TripletBuilder class, which transforms segmented 
context-response pairs into a structured triplet format (Anchor, Positive, Negative). 
While the current dataset provides only positive responses, this schema is 
designed for forward compatibility with contrastive learning techniques and 
retrieval-augmented generation scenarios requiring negative sampling.
"""

import logging
from typing import List, Dict, Optional
from config import Config


class TripletBuilder:
    """
    Constructs triplet structures from context-response dialogue pairs.

    Attributes:
        config: An instance of the Config dataclass containing pipeline settings.
        logger: A standard Python logger for tracking the triplet building process.
    """

    def __init__(self, config: Config) -> None:
        """
        Initializes the TripletBuilder with the required configuration.

        Args:
            config: Configuration object defining pipeline parameters.
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

    def build_triplets(
        self, 
        dialogue_pairs: List[Dict[str, str]]
    ) -> List[Dict[str, Optional[str]]]:
        """
        Converts a list of context-response pairs into a list of triplets.

        The 'context' becomes the 'anchor', and the 'response' becomes the 'positive'.
        The 'negative' field is initialized to None to allow future extensibility.

        Args:
            dialogue_pairs: A list of dictionaries, each containing 'context' and 
                            'response' string keys.

        Returns:
            List[Dict[str, Optional[str]]]: A list of dictionaries with keys 
            'anchor', 'positive', and 'negative'.

        Raises:
            TypeError: If the input is not a list or contains invalid element types.
            KeyError: If a dictionary in the input list is missing required keys.
        """
        if not isinstance(dialogue_pairs, list):
            self.logger.error("Input dialogue_pairs must be a list.")
            raise TypeError("Expected a list of dictionaries for dialogue_pairs.")

        self.logger.info("Starting triplet construction for %d pairs.", len(dialogue_pairs))
        triplets: List[Dict[str, Optional[str]]] = []

        for index, pair in enumerate(dialogue_pairs):
            if not isinstance(pair, dict):
                self.logger.error("Element at index %d is not a dictionary.", index)
                raise TypeError(f"Element at index {index} must be a dictionary.")

            if "context" not in pair or "response" not in pair:
                self.logger.error(
                    "Missing 'context' or 'response' key in pair at index %d.", index
                )
                raise KeyError(
                    f"Pair at index {index} is missing 'context' or 'response'."
                )

            triplet: Dict[str, Optional[str]] = {
                "anchor": pair["context"],
                "positive": pair["response"],
                "negative": None  # Placeholder for future negative mining implementations
            }
            triplets.append(triplet)

        self.logger.info("Successfully constructed %d triplets.", len(triplets))
        return triplets