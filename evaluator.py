"""
Evaluation module for the Telugu Multi-Turn Dialogue Research Pipeline.

This module provides the Evaluator class, which applies similarity metrics
to compare generated responses against ground-truth positive responses.
Supports checkpointing for crash-safe resumability during long evaluation runs.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Set
import numpy as np
from tqdm import tqdm

from config import Config
from similarity_metrics import SimilarityMetrics
from embedder import MuRILEmbedder


class Evaluator:
    """
    Evaluates generated dialogue responses against ground-truth positive responses.

    Attributes:
        config: An instance of the Config dataclass containing pipeline settings.
        logger: A standard Python logger for tracking evaluation events.
        metrics: An instance of SimilarityMetrics for computing similarities.
        embedder: An instance of MuRILEmbedder for batch generating dense text vectors.
    """

    def __init__(self, config: Config) -> None:
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
        Computes Cosine, Jaccard, Dice, and BERTScore F1 for all generated candidates.
        Saves checkpoint every 500 rows and resumes from last checkpoint on restart.
        """
        if not isinstance(enriched_triplets, list):
            raise TypeError("Expected a list of enriched triplets.")
        if not enriched_triplets:
            raise ValueError("The input dataset cannot be empty.")

        self.logger.info("Starting evaluation for %d dialogue items.", len(enriched_triplets))
        
        # Step 1: Collect all unique texts for embedding
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
            self.logger.error("Failed to generate embeddings: %s", str(e))
            raise RuntimeError("Batch embedding failed in the evaluator.") from e

        text_to_embedding: Dict[str, np.ndarray] = {
            text: embeddings[i] for i, text in enumerate(text_list)
        }

        # Step 3: Load evaluation checkpoint if it exists
        evaluation_results: List[Dict[str, Any]] = []
        start_index = 0

        if os.path.exists(self.config.evaluation_checkpoint_file):
            try:
                with open(self.config.evaluation_checkpoint_file, "r", encoding="utf-8") as f:
                    evaluation_results = json.load(f)
                start_index = len(evaluation_results)
                self.logger.info(
                    "Loaded evaluation checkpoint. Resuming from index %d of %d.",
                    start_index, len(enriched_triplets)
                )
            except Exception as e:
                self.logger.warning("Failed to load eval checkpoint: %s. Starting fresh.", str(e))
                evaluation_results = []
                start_index = 0

        # Step 4: Compute pairwise metrics with checkpointing and tqdm
        for index in tqdm(
            range(start_index, len(enriched_triplets)),
            desc="Evaluating Responses",
            unit="row",
            initial=start_index,
            total=len(enriched_triplets)
        ):
            row = enriched_triplets[index]
            positive_text = row.get("positive")
            if not isinstance(positive_text, str) or not positive_text.strip():
                continue

            positive_vector = text_to_embedding.get(positive_text)
            if positive_vector is None:
                continue
            
            row_eval: Dict[str, Any] = {
                "anchor": row.get("anchor"),
                "positive": positive_text
            }

            for key in generation_keys:
                candidate_text = row.get(key)
                cos_sim, jac_sim, dice_sim, bert_f1 = 0.0, 0.0, 0.0, 0.0
                
                if isinstance(candidate_text, str) and candidate_text.strip():
                    candidate_vector = text_to_embedding.get(candidate_text)
                    
                    try:
                        if candidate_vector is not None:
                            cos_sim = self.metrics.compute_cosine_similarity(
                                positive_vector, candidate_vector
                            )
                        jac_sim = self.metrics.compute_jaccard_similarity(
                            positive_text, candidate_text
                        )
                        dice_sim = self.metrics.compute_dice_similarity(
                            positive_text, candidate_text
                        )
                        _, _, bert_f1 = self.metrics.compute_bert_score(
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
                row_eval[f"{key}_bert_f1"] = bert_f1

            evaluation_results.append(row_eval)

            # Save checkpoint every 500 rows
            if len(evaluation_results) % 500 == 0:
                self.logger.info("Saving evaluation checkpoint at row %d...", len(evaluation_results))
                os.makedirs(os.path.dirname(self.config.evaluation_checkpoint_file), exist_ok=True)
                with open(self.config.evaluation_checkpoint_file, "w", encoding="utf-8") as f:
                    json.dump(evaluation_results, f, ensure_ascii=False, indent=2)

        # Final save
        os.makedirs(os.path.dirname(self.config.evaluation_checkpoint_file), exist_ok=True)
        with open(self.config.evaluation_checkpoint_file, "w", encoding="utf-8") as f:
            json.dump(evaluation_results, f, ensure_ascii=False, indent=2)

        self.logger.info("Successfully evaluated %d dialogue rows.", len(evaluation_results))
        return evaluation_results