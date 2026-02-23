"""
Evaluation module for the Telugu Multi-Turn Dialogue Research Pipeline.

This module provides the Evaluator class, which applies similarity metrics
to compare generated responses against ground-truth positive responses.
It handles the batch embedding of all texts using MuRIL and constructs a 
comprehensive evaluation record for every dialogue turn across all models 
and prompting strategies.
"""

import logging
from typing import List, Dict, Any, Optional, Set
import numpy as np

from config import Config
from similarity_metrics import SimilarityMetrics
from embedder import MuRILEmbedder


class Evaluator:
    """
    Evaluates generated dialogue responses against ground-truth positive responses.

    Attributes:
        config: An instance of the Config dataclass containing pipeline settings.
        logger: A standard Python logger for tracking evaluation events.
        metrics: An instance of SimilarityMetrics for computing string and vector similarities.
        embedder: An instance of MuRILEmbedder for batch generating dense text vectors.
    """

    def __init__(self, config: Config) -> None:
        """
        Initializes the Evaluator, along with required metric calculators and embedders.

        Args:
            config: Configuration object defining model settings and parameters.

        Raises:
            RuntimeError: If embedder or metric classes fail to initialize.
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("Initializing Evaluator components.")
        try:
            self.metrics = SimilarityMetrics(config)
            self.embedder = MuRILEmbedder(config)
        except Exception as e:
            self.logger.error("Failed to initialize evaluation components: %s", str(e))
            raise RuntimeError("Could not set up SimilarityMetrics or MuRILEmbedder.") from e

    def evaluate_dataset(
        self, 
        enriched_triplets: List[Dict[str, Optional[str]]]
    ) -> List[Dict[str, Any]]:
        """
        Computes Cosine, Jaccard, and Dice similarities for all generated candidates.

        To optimize processing time, this method extracts all unique strings
        (both ground truth and generated candidates) and performs a single batch 
        embedding run before computing pairwise similarities.

        Args:
            enriched_triplets: A list of dictionaries containing 'anchor', 'positive', 
                               and the generated response keys ('t5_raw', 't5_cot', 
                               'sarvam_raw', 'sarvam_cot').

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing the original anchor,
                                  positive text, candidate texts, and all computed metrics.

        Raises:
            TypeError: If the input dataset is not a list.
            ValueError: If the input dataset is empty.
        """
        if not isinstance(enriched_triplets, list):
            self.logger.error("Input dataset must be a list.")
            raise TypeError("Expected a list of enriched triplets.")

        if not enriched_triplets:
            self.logger.error("Received empty list for evaluation.")
            raise ValueError("The input dataset cannot be empty.")

        self.logger.info("Starting evaluation for %d dialogue items.", len(enriched_triplets))
        
        # Step 1: Collect all unique texts required for embedding
        unique_texts: Set[str] = set()
        generation_keys = ["t5_raw", "t5_cot", "sarvam_raw"]
        
        for row in enriched_triplets:
            if isinstance(row.get("positive"), str):
                unique_texts.add(row["positive"])
            for key in generation_keys:
                candidate = row.get(key)
                if isinstance(candidate, str) and candidate.strip():
                    unique_texts.add(candidate)

        text_list = list(unique_texts)
        if not text_list:
            self.logger.warning("No valid text found to embed and evaluate.")
            return []

        self.logger.info("Batch embedding %d unique text strings.", len(text_list))
        
        # Step 2: Batch embed all unique texts
        try:
            embeddings = self.embedder.get_embeddings(text_list)
        except Exception as e:
            self.logger.error("Failed to generate embeddings during evaluation: %s", str(e))
            raise RuntimeError("Batch embedding failed in the evaluator.") from e

        # Map text back to its embedding vector
        text_to_embedding: Dict[str, np.ndarray] = {
            text: embeddings[i] for i, text in enumerate(text_list)
        }

        # Step 3: Compute pairwise metrics
        evaluation_results: List[Dict[str, Any]] = []

        for index, row in enumerate(enriched_triplets):
            positive_text = row.get("positive")
            if not isinstance(positive_text, str) or not positive_text.strip():
                self.logger.debug("Skipping row %d due to missing positive text.", index)
                continue

            positive_vector = text_to_embedding[positive_text]
            
            row_eval: Dict[str, Any] = {
                "anchor": row.get("anchor"),
                "positive": positive_text
            }

            for key in generation_keys:
                candidate_text = row.get(key)
                
                # Default scores if generation failed or is empty
                cos_sim, jac_sim, dice_sim = 0.0, 0.0, 0.0
                
                if isinstance(candidate_text, str) and candidate_text.strip():
                    candidate_vector = text_to_embedding[candidate_text]
                    
                    try:
                        cos_sim = self.metrics.compute_cosine_similarity(
                            positive_vector, candidate_vector
                        )
                        jac_sim = self.metrics.compute_jaccard_similarity(
                            positive_text, candidate_text
                        )
                        dice_sim = self.metrics.compute_dice_similarity(
                            positive_text, candidate_text
                        )
                    except Exception as e:
                        self.logger.warning(
                            "Error computing metrics for row %d, key %s: %s", 
                            index, key, str(e)
                        )

                row_eval[f"{key}_text"] = candidate_text
                row_eval[f"{key}_cosine"] = cos_sim
                row_eval[f"{key}_jaccard"] = jac_sim
                row_eval[f"{key}_dice"] = dice_sim

            evaluation_results.append(row_eval)

        self.logger.info("Successfully evaluated %d dialogue rows.", len(evaluation_results))
        return evaluation_results