"""
Response generation orchestration module for the Telugu Multi-Turn Dialogue Pipeline.

This module provides the ResponseGenerator class, which coordinates the generation
of response candidates across multiple models (IndicT5 and Sarvam) and prompting
strategies (Raw and Chain-of-Thought). It aggregates the generated outputs into
a unified structure for downstream embedding and evaluation.
"""

import os
import json
import logging
from typing import List, Dict, Optional
from tqdm import tqdm

from config import Config
from model_t5 import IndicT5Model
from model_sarvam import SarvamModel
from prompt_builder import PromptBuilder


class ResponseGenerator:
    """
    Coordinates multi-model and multi-prompt response generation.

    Attributes:
        config: An instance of the Config dataclass containing pipeline settings.
        logger: A standard Python logger for tracking the generation process.
        prompt_builder: An instance of PromptBuilder for creating model inputs.
        t5_model: An initialized IndicT5Model instance.
        sarvam_model: An initialized SarvamModel instance.
    """

    def __init__(self, config: Config) -> None:
        """
        Initializes the ResponseGenerator and loads the required models in memory.

        Args:
            config: Configuration object defining model settings and generation parameters.

        Raises:
            RuntimeError: If model initialization fails due to hardware or memory issues.
            OSError: If model weights cannot be downloaded or found.
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("Initializing PromptBuilder and Generation Models.")
        self.prompt_builder = PromptBuilder(config)
        
        # Models are loaded here. Depending on hardware, this may require significant VRAM.
        self.t5_model = IndicT5Model(config)
        self.sarvam_model = SarvamModel(config)
        
        self.logger.info("All generation models initialized successfully.")

    def generate_all_responses(
        self, 
        triplets: List[Dict[str, Optional[str]]]
    ) -> List[Dict[str, Optional[str]]]:
        """
        Generates four distinct responses for each dialogue context in the input.

        For every triplet, this method uses the 'anchor' (context) to generate:
        1. IndicT5 Raw Response
        2. IndicT5 Chain-of-Thought (COT) Response
        3. Sarvam Raw Response
        4. Sarvam Chain-of-Thought (COT) Response

        The generated responses are appended to the original triplet dictionary.

        Args:
            triplets: A list of dictionaries, each containing an 'anchor' string.

        Returns:
            List[Dict[str, Optional[str]]]: The enriched list of dictionaries containing 
            the original triplet data plus the four generated response candidates.

        Raises:
            TypeError: If the input is not a list or contains invalid elements.
            KeyError: If a triplet dictionary is missing the 'anchor' key.
        """
        if not isinstance(triplets, list):
            self.logger.error("Input triplets must be a list.")
            raise TypeError("Expected a list of dictionaries for triplets.")

        self.logger.info("Starting response generation for %d context windows.", len(triplets))
        
        enriched_results: List[Dict[str, Optional[str]]] = []
        start_index = 0

        # Load Checkpoint if it exists
        if os.path.exists(self.config.checkpoint_file):
            try:
                with open(self.config.checkpoint_file, "r", encoding="utf-8") as f:
                    enriched_results = json.load(f)
                start_index = len(enriched_results)
                self.logger.info("Loaded checkpoint. Resuming from index %d out of %d.", start_index, len(triplets))
            except Exception as e:
                self.logger.warning("Failed to load checkpoint, starting from scratch: %s", str(e))
                enriched_results = []
                start_index = 0

        # We need the number of remaining batches based on the checkpoint
        total_batches = (len(triplets) - start_index + self.config.generation_batch_size - 1) // self.config.generation_batch_size
        
        # Generator to handle the batches
        def batch_generator():
            for batch_start in range(start_index, len(triplets), self.config.generation_batch_size):
                batch_end = min(batch_start + self.config.generation_batch_size, len(triplets))
                yield batch_start, batch_end, triplets[batch_start:batch_end]

        self.logger.info("Initializing tqdm progress bar for %d batches.", total_batches)
        
        for batch_start, batch_end, batch_triplets in tqdm(batch_generator(), total=total_batches, desc="Generating Responses", unit="batch"):
            self.logger.debug("Processing batch %d to %d (out of %d).", batch_start + 1, batch_end, len(triplets))

            raw_prompts = []
            cot_prompts = []
            valid_triplets = []

            for index_in_batch, triplet in enumerate(batch_triplets):
                absolute_index = batch_start + index_in_batch
                
                if not isinstance(triplet, dict):
                    self.logger.error("Element at index %d is not a dictionary.", absolute_index)
                    raise TypeError(f"Element at index {absolute_index} must be a dictionary.")

                if "anchor" not in triplet or triplet["anchor"] is None:
                    self.logger.error("Missing or null 'anchor' key in triplet at index %d.", absolute_index)
                    raise KeyError(f"Triplet at index {absolute_index} is missing the 'anchor' context.")

                context = str(triplet["anchor"])
                raw_prompts.append(self.prompt_builder.build_raw_prompt(context))
                cot_prompts.append(self.prompt_builder.build_cot_prompt(context))
                valid_triplets.append(triplet)

            if not valid_triplets:
                continue

            try:
                # Generate Batches
                t5_raw_responses = self.t5_model.generate_responses(raw_prompts)
                t5_cot_responses = self.t5_model.generate_responses(cot_prompts)
                sarvam_raw_responses = self.sarvam_model.generate_responses(raw_prompts)

                for i, triplet in enumerate(valid_triplets):
                    enriched_triplet = triplet.copy()
                    enriched_triplet["t5_raw"] = t5_raw_responses[i]
                    enriched_triplet["t5_cot"] = t5_cot_responses[i]
                    enriched_triplet["sarvam_raw"] = sarvam_raw_responses[i]
                    enriched_results.append(enriched_triplet)

            except Exception as e:
                self.logger.error(
                    "Batch generation failed from index %d to %d. Error: %s. Filling with None.", 
                    batch_start, batch_end, str(e)
                )
                for triplet in valid_triplets:
                    enriched_triplet = triplet.copy()
                    enriched_triplet["t5_raw"] = None
                    enriched_triplet["t5_cot"] = None
                    enriched_triplet["sarvam_raw"] = None
                    enriched_results.append(enriched_triplet)

            # Save checkpoing periodically (every N batches based on interval)
            # interval 100 means save every 100 items generated. 
            if len(enriched_results) % self.config.checkpoint_interval < self.config.generation_batch_size:
                self.logger.info("Saving generation checkpoint at triplet %d...", len(enriched_results))
                os.makedirs(os.path.dirname(self.config.checkpoint_file), exist_ok=True)
                with open(self.config.checkpoint_file, "w", encoding="utf-8") as f:
                    json.dump(enriched_results, f, ensure_ascii=False, indent=2)

        # Final save to ensure the last batch is committed
        os.makedirs(os.path.dirname(self.config.checkpoint_file), exist_ok=True)
        with open(self.config.checkpoint_file, "w", encoding="utf-8") as f:
            json.dump(enriched_results, f, ensure_ascii=False, indent=2)

        self.logger.info("Completed response generation for all context windows.")
        return enriched_results