"""
Central configuration module for the Telugu Multi-Turn Dialogue Research Pipeline.

Provides a frozen dataclass containing all file paths, model identifiers,
generation hyperparameters, and processing settings used across the pipeline.
"""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    """
    Immutable configuration object for the Telugu dialogue pipeline.

    All directory paths are resolved relative to the project root,
    which is determined from the location of this configuration file.
    """

    project_root: str = os.path.dirname(os.path.abspath(__file__))

    # Dataset paths (pre-split IndicDialogue Telugu subset)
    dataset_root: str = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "IndicDialogue Dataset", "dataset"
    )
    train_data_path: str = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "IndicDialogue Dataset", "dataset", "IndicDialogue", "Telugu", "Telugu.jsonl"
    )
    test_data_path: str = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "IndicDialogue Dataset", "dataset", "IndicDialogue", "Telugu", "Telugu.jsonl"
    )

    # Output directories
    output_dir: str = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "outputs"
    )
    logs_dir: str = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "logs"
    )
    vector_db_path: str = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "vector_db"
    )

    # Dialogue segmentation
    context_window_size: int = 5

    # Telugu purity filtering
    telugu_purity_threshold: float = 1.0

    # Embedding model (MuRIL)
    embedding_model_name: str = "google/muril-base-cased"
    embedding_batch_size: int = 64   # RTX6000 Ada 48GB: doubled from 32

    # Gemma generation model
    t5_model_name: str = "google/gemma-3-1b-it"

    # Sarvam generation model
    sarvam_model_name: str = "sarvamai/sarvam-1"

    # Generation hyperparameters (tuned for RTX6000 Ada — 48GB VRAM, Ada Lovelace sm_89)
    generation_batch_size: int = 8   # 48GB handles both Sarvam(14GB)+Gemma(2.5GB)+KV-cache comfortably
    max_length: int = 256
    num_beams: int = 1               # greedy decoding — fastest; set to 4 for beam search
    temperature: float = 0.7
    top_p: float = 0.9

    # Dataset sampling
    max_samples: int = 10000   # Max triplets to process (stratified across movies); -1 = use all

    # Generation Checkpointing
    checkpoint_file: str = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "outputs", "generation_checkpoint.json"
    )
    checkpoint_interval: int = 100

    # Evaluation Checkpointing
    evaluation_checkpoint_file: str = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "outputs", "evaluation_checkpoint.json"
    )

    # Embedding Checkpointing
    embedding_checkpoint_file: str = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "outputs", "embedding_checkpoint.npz"
    )