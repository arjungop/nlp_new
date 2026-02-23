"""
Main execution pipeline for the Telugu Multi-Turn Dialogue Research Project.

This module serves as the entry point for the end-to-end processing pipeline.
It orchestrates data loading, cleaning, segmentation, triplet construction,
embedding, vector storage, multi-model response generation, evaluation,
and final visualization.
"""

import os
import sys
import logging
from typing import List, Dict, Optional

from config import Config
from data_loader import DataLoader
from data_cleaner import DataCleaner
from dialogue_segmenter import DialogueSegmenter
from tripplet_builder import TripletBuilder
from embedder import MuRILEmbedder
from vectordb_store import VectorDBStore
from response_generator import ResponseGenerator
from evaluator import Evaluator
from metrics_logger import MetricsLogger
from results_analyzer import ResultsAnalyzer
from visualizer import Visualizer


def setup_logging(config: Config) -> None:
    """
    Configures the root logger for the entire pipeline application.

    Creates the logging directory if it does not exist and sets up both
    console and file handlers with a standardized formatting layout.

    Args:
        config: Configuration object containing the logs_dir path.

    Raises:
        OSError: If the logging directory cannot be created.
    """
    try:
        os.makedirs(config.logs_dir, exist_ok=True)
    except OSError as e:
        print(f"CRITICAL: Failed to create logging directory {config.logs_dir}. Error: {e}")
        sys.exit(1)

    log_file = os.path.join(config.logs_dir, "pipeline.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="w", encoding="utf-8"),
            logging.StreamHandler(sys.stdout)
        ]
    )


def main() -> None:
    """
    Executes the complete Telugu dialogue processing and evaluation pipeline.

    Initializes all components sequentially, manages data handoffs between
    modules, and catches top-level execution errors to ensure safe termination.

    Raises:
        SystemExit: If a critical pipeline stage fails, execution is halted.
    """
    config = Config()
    setup_logging(config)
    logger = logging.getLogger("Pipeline")

    print("Starting Telugu Multi-Turn Dialogue Pipeline...")
    logger.info("Pipeline initialized with context window size: %d", config.context_window_size)

    try:
        # Data Loading
        print("Loading Telugu dialogue data...")
        loader = DataLoader(config)
        raw_df = loader.load_data(split="train")

        # Data Cleaning
        print("Cleaning dataset...")
        cleaner = DataCleaner(config)
        cleaned_df = cleaner.clean_dataset(raw_df, text_column="text")

        # Dialogue Segmentation
        print("Segmenting dialogues into multi-turn context windows...")
        segmenter = DialogueSegmenter(config)
        dialogue_pairs = segmenter.segment_dialogues(cleaned_df, text_column="text")

        if not dialogue_pairs:
            logger.error("No dialogue pairs generated. Halting pipeline.")
            print("Error: Segmentation resulted in zero pairs.")
            sys.exit(1)

        # Triplet Construction
        print("Building Anchor-Positive triplets...")
        builder = TripletBuilder(config)
        triplets = builder.build_triplets(dialogue_pairs)

        # Embedding and Vector Database Storage
        print("Embedding ground-truth contexts using MuRIL...")
        embedder = MuRILEmbedder(config)

        contexts = [str(t["anchor"]) for t in triplets if t["anchor"]]
        context_embeddings = embedder.get_embeddings(contexts)

        print("Storing embeddings in FAISS Vector DB...")
        vector_db = VectorDBStore(config)
        vector_db.build_index(context_embeddings, triplets)
        vector_db.save_index()

        # Response Generation (T5 and Sarvam, Raw and COT)
        print("Generating responses (this may take a while depending on hardware)...")
        generator = ResponseGenerator(config)
        enriched_triplets = generator.generate_all_responses(triplets)

        # Evaluation against Ground Truth
        print("Evaluating generated responses against positive targets...")
        evaluator = Evaluator(config)
        evaluation_results = evaluator.evaluate_dataset(enriched_triplets)

        # Logging Metrics
        print("Saving evaluation results to disk...")
        metrics_logger = MetricsLogger(config)
        metrics_logger.log_results(evaluation_results)

        # Results Analysis
        print("Analyzing performance metrics...")
        analyzer = ResultsAnalyzer(config)
        summary_matrix = analyzer.analyze_metrics(evaluation_results)
        analyzer.print_summary(summary_matrix)

        # Visualization
        print("Generating visual plots (Heatmap and Bar Charts)...")
        visualizer = Visualizer(config)
        visualizer.plot_heatmap(summary_matrix)
        visualizer.plot_bar_charts(summary_matrix)

        print("Pipeline execution completed successfully.")
        logger.info("Pipeline finished without errors. Outputs saved to %s", config.output_dir)

    except FileNotFoundError as e:
        logger.critical("A required file was not found: %s", str(e))
        print(f"CRITICAL ERROR: File not found - {e}")
        sys.exit(1)
    except OSError as e:
        logger.critical("An OS or I/O error occurred: %s", str(e))
        print(f"CRITICAL ERROR: I/O failure - {e}")
        sys.exit(1)
    except RuntimeError as e:
        logger.critical("A runtime execution error occurred: %s", str(e))
        print(f"CRITICAL ERROR: Runtime failure - {e}")
        sys.exit(1)
    except Exception as e:
        logger.critical("An unexpected error halted the pipeline: %s", str(e), exc_info=True)
        print(f"CRITICAL ERROR: Unexpected failure - {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()