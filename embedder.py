"""
Embedding module for the Telugu Multi-Turn Dialogue Research Pipeline.

Provides the MuRILEmbedder class, which generates dense vector
representations of Telugu text using the MuRIL (Multilingual
Representations for Indian Languages) transformer model. Embeddings
are computed as mean-pooled last hidden states with attention masking.
"""

import logging
from typing import List

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

from config import Config


class MuRILEmbedder:
    """
    Generates dense embeddings for Telugu text using MuRIL.

    Attributes:
        config: Pipeline configuration object.
        logger: Module-level logger.
        device: PyTorch compute device.
        tokenizer: MuRIL tokenizer.
        model: MuRIL transformer model.
    """

    def __init__(self, config: Config) -> None:
        """
        Initializes the MuRILEmbedder, loading the tokenizer and model.

        Args:
            config: Configuration object containing embedding_model_name
                    and embedding_batch_size.

        Raises:
            OSError: If the model cannot be found or downloaded.
            RuntimeError: If device allocation fails.
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.logger.info(
            "Initializing MuRILEmbedder with model: %s on %s",
            self.config.embedding_model_name, self.device
        )

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.embedding_model_name
            )
            self.model = AutoModel.from_pretrained(
                self.config.embedding_model_name
            )
            self.model.to(self.device)
            self.model.eval()
        except OSError as e:
            self.logger.error(
                "Failed to load MuRIL model or tokenizer: %s", str(e)
            )
            raise OSError(
                f"Could not load model {self.config.embedding_model_name}."
            ) from e
        except RuntimeError as e:
            self.logger.error(
                "Hardware error while loading MuRIL model: %s", str(e)
            )
            raise RuntimeError(
                "Failed to allocate MuRIL model to compute device."
            ) from e

    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Computes dense embeddings for a list of text strings.

        Texts are processed in batches of config.embedding_batch_size.
        Each embedding is the mean of the last hidden state vectors,
        weighted by the attention mask to exclude padding tokens.

        Args:
            texts: A list of strings to embed.

        Returns:
            A 2D numpy array of shape (len(texts), hidden_dim).

        Raises:
            ValueError: If the input list is empty.
            TypeError: If the input is not a list of strings.
        """
        if not isinstance(texts, list):
            raise TypeError("Expected a list of strings.")
        if not texts:
            raise ValueError("Cannot embed an empty list of texts.")

        batch_size = self.config.embedding_batch_size
        all_embeddings: List[np.ndarray] = []

        self.logger.info(
            "Embedding %d texts in batches of %d.", len(texts), batch_size
        )

        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            batch_embeddings = self._embed_batch(batch)
            all_embeddings.append(batch_embeddings)

        result = np.concatenate(all_embeddings, axis=0)
        self.logger.info(
            "Embedding complete. Output shape: %s", result.shape
        )

        return result

    def _embed_batch(self, batch: List[str]) -> np.ndarray:
        """
        Computes mean-pooled embeddings for a single batch.

        Args:
            batch: A list of strings forming one processing batch.

        Returns:
            A 2D numpy array of shape (len(batch), hidden_dim).
        """
        encoded = self.tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        last_hidden = outputs.last_hidden_state
        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
        sum_embeddings = torch.sum(last_hidden * mask_expanded, dim=1)
        sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
        mean_pooled = sum_embeddings / sum_mask

        return mean_pooled.cpu().numpy()