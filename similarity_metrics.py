"""
Similarity metrics module for the Telugu Multi-Turn Dialogue Research Pipeline.

This module provides the SimilarityMetrics class, which calculates various
similarity scores between generated responses and the ground-truth positive
responses. It supports Cosine similarity for dense vectors (MuRIL embeddings),
Jaccard/Dice similarities for subword token sets, and BERTScore for
semantically-aware evaluation using contextual embeddings.
"""

import logging
from typing import Set, Tuple
import numpy as np
from transformers import AutoTokenizer
from bert_score import score as bert_score_fn
from config import Config


class SimilarityMetrics:
    """
    Computes text and vector similarity metrics for response evaluation.

    Attributes:
        config: An instance of the Config dataclass containing pipeline settings.
        logger: A standard Python logger for tracking metric calculations.
        tokenizer: The tokenizer used to split text into subword token sets for 
                   Jaccard and Dice similarity calculations.
    """

    def __init__(self, config: Config) -> None:
        """
        Initializes the SimilarityMetrics class and loads the subword tokenizer.

        Args:
            config: Configuration object containing the embedding_model_name 
                    for tokenizer initialization.

        Raises:
            OSError: If the tokenizer cannot be loaded from the configured model name.
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("Initializing SimilarityMetrics and loading tokenizer: %s", 
                         self.config.embedding_model_name)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.embedding_model_name)
        except OSError as e:
            self.logger.error("Failed to load tokenizer for metric evaluation: %s", str(e))
            raise OSError(f"Could not load tokenizer {self.config.embedding_model_name}.") from e

    def compute_bert_score(self, reference: str, candidate: str) -> Tuple[float, float, float]:
        """
        Computes BERTScore (Precision, Recall, F1) between a reference and candidate text.

        BERTScore leverages contextual embeddings from a pretrained model to compute
        token-level similarity, making it far more semantically meaningful than
        surface-level overlap metrics like Jaccard or Dice. It uses MuRIL
        (google/muril-base-cased) as the underlying model, which is specifically
        designed for Indian languages including Telugu.

        Args:
            reference: The ground-truth reference text string.
            candidate: The generated candidate text string.

        Returns:
            Tuple[float, float, float]: A tuple of (Precision, Recall, F1) scores,
            each ranging from 0.0 to 1.0.

        Raises:
            TypeError: If either input is not a string.
        """
        if not isinstance(reference, str) or not isinstance(candidate, str):
            self.logger.error("BERTScore inputs must be strings.")
            raise TypeError("Expected strings for BERTScore computation.")

        if not reference.strip() or not candidate.strip():
            self.logger.warning("Empty input to BERTScore. Returning zeros.")
            return 0.0, 0.0, 0.0

        try:
            precision, recall, f1 = bert_score_fn(
                cands=[candidate],
                refs=[reference],
                model_type=self.config.embedding_model_name,
                num_layers=12,
                verbose=False,
                device=None,  # auto-detect GPU/CPU
            )
            return (
                float(precision[0].item()),
                float(recall[0].item()),
                float(f1[0].item()),
            )
        except Exception as e:
            self.logger.warning("BERTScore computation failed: %s. Returning zeros.", str(e))
            return 0.0, 0.0, 0.0

    def compute_cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculates the cosine similarity between two 1D numpy arrays.

        Args:
            vec1: The first embedding vector (1D numpy array).
            vec2: The second embedding vector (1D numpy array).

        Returns:
            float: The cosine similarity score, ranging from -1.0 to 1.0. 
                   Returns 0.0 if either vector has a norm of zero.

        Raises:
            TypeError: If the inputs are not numpy arrays.
            ValueError: If the inputs are not 1D arrays or have mismatched shapes.
        """
        if not isinstance(vec1, np.ndarray) or not isinstance(vec2, np.ndarray):
            self.logger.error("Cosine similarity inputs must be numpy arrays.")
            raise TypeError("Expected numpy arrays for vector similarity.")

        if vec1.ndim != 1 or vec2.ndim != 1:
            self.logger.error("Input vectors must be 1-dimensional.")
            raise ValueError("Cosine similarity requires 1D numpy arrays.")

        if vec1.shape != vec2.shape:
            self.logger.error("Shape mismatch: %s vs %s", vec1.shape, vec2.shape)
            raise ValueError("Input vectors must have the same shape.")

        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0.0 or norm2 == 0.0:
            self.logger.warning("One of the vectors has a zero norm. Returning similarity 0.0.")
            return 0.0

        similarity = np.dot(vec1, vec2) / (norm1 * norm2)
        return float(similarity)

    def _get_token_set(self, text: str) -> Set[str]:
        """
        Converts a text string into a set of subword tokens.

        Args:
            text: The input string to tokenize.

        Returns:
            Set[str]: A set containing the unique subword tokens.
            
        Raises:
            TypeError: If the input text is not a string.
        """
        if not isinstance(text, str):
            self.logger.error("Input to tokenization must be a string.")
            raise TypeError("Expected a string for tokenization.")
            
        tokens = self.tokenizer.tokenize(text)
        return set(tokens)

    def compute_jaccard_similarity(self, text1: str, text2: str) -> float:
        """
        Calculates the Jaccard similarity between two strings based on subword tokens.

        Jaccard Similarity = (Intersection of sets) / (Union of sets)

        Args:
            text1: The first text string.
            text2: The second text string.

        Returns:
            float: The Jaccard similarity score, ranging from 0.0 to 1.0.

        Raises:
            TypeError: If either input is not a string.
        """
        set1 = self._get_token_set(text1)
        set2 = self._get_token_set(text2)

        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))

        if union == 0:
            return 0.0

        return float(intersection) / float(union)

    def compute_dice_similarity(self, text1: str, text2: str) -> float:
        """
        Calculates the Dice similarity (Sorensen-Dice coefficient) between two strings.

        Dice Similarity = (2 * Intersection of sets) / (Size of Set 1 + Size of Set 2)

        Args:
            text1: The first text string.
            text2: The second text string.

        Returns:
            float: The Dice similarity score, ranging from 0.0 to 1.0.

        Raises:
            TypeError: If either input is not a string.
        """
        set1 = self._get_token_set(text1)
        set2 = self._get_token_set(text2)

        intersection = len(set1.intersection(set2))
        total_elements = len(set1) + len(set2)

        if total_elements == 0:
            return 0.0

        return (2.0 * float(intersection)) / float(total_elements)