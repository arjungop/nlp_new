"""
Vector database module for the Telugu Multi-Turn Dialogue Research Pipeline.

This module provides the VectorDBStore class, which acts as a wrapper around
FAISS (Facebook AI Similarity Search). It handles the indexing of MuRIL 
embeddings, storage of associated triplet metadata, persistence to disk, 
and efficient similarity-based retrieval.
"""

import os
import pickle
import logging
from typing import List, Dict, Optional, Tuple
import numpy as np
import faiss
from config import Config


class VectorDBStore:
    """
    Manages the creation, persistence, and querying of a FAISS vector index.

    Attributes:
        config: An instance of the Config dataclass containing paths and settings.
        logger: A standard Python logger for tracking vector DB operations.
        index: The underlying FAISS index object.
        metadata: A list of dictionaries corresponding to the vectors in the index.
    """

    def __init__(self, config: Config) -> None:
        """
        Initializes the VectorDBStore.

        Args:
            config: Configuration object containing vector_db_path.
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.index: Optional[faiss.Index] = None
        self.metadata: List[Dict[str, Optional[str]]] = []

    def build_index(
        self, 
        embeddings: np.ndarray, 
        metadata: List[Dict[str, Optional[str]]]
    ) -> None:
        """
        Builds a new FAISS IndexFlatL2 from the provided embeddings and metadata.

        Args:
            embeddings: A 2D numpy array of shape (num_samples, embedding_dim).
            metadata: A list of dictionaries containing triplet data. Must match
                      the number of rows in the embeddings array.

        Raises:
            ValueError: If the number of embeddings does not match the metadata length,
                        or if the embeddings array is empty.
            TypeError: If the embeddings are not a numpy array.
        """
        if not isinstance(embeddings, np.ndarray):
            self.logger.error("Embeddings must be a numpy array.")
            raise TypeError("Expected embeddings to be a numpy array.")

        if embeddings.size == 0 or embeddings.ndim != 2:
            self.logger.error("Embeddings array is empty or not 2D.")
            raise ValueError("Embeddings must be a non-empty 2D numpy array.")

        if len(embeddings) != len(metadata):
            self.logger.error(
                "Dimension mismatch: %d embeddings vs %d metadata items.",
                len(embeddings), len(metadata)
            )
            raise ValueError("Number of embeddings must match the length of metadata.")

        dimension: int = embeddings.shape[1]
        self.logger.info("Building FAISS IndexFlatL2 with dimension: %d", dimension)
        
        # Ensure embeddings are float32 as required by FAISS
        embeddings_float32 = np.ascontiguousarray(embeddings, dtype=np.float32)
        
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings_float32)
        self.metadata = metadata
        
        self.logger.info("Successfully added %d vectors to the index.", self.index.ntotal)

    def save_index(self) -> None:
        """
        Persists the FAISS index and associated metadata to the filesystem.

        Creates the target directory if it does not exist.

        Raises:
            RuntimeError: If the index has not been built or loaded yet.
            IOError: If saving the index or metadata files fails.
        """
        if self.index is None:
            self.logger.error("Attempted to save an uninitialized index.")
            raise RuntimeError("Cannot save: Index is not initialized.")

        os.makedirs(self.config.vector_db_path, exist_ok=True)
        
        index_file = os.path.join(self.config.vector_db_path, "faiss.index")
        metadata_file = os.path.join(self.config.vector_db_path, "metadata.pkl")

        self.logger.info("Saving FAISS index to %s", index_file)
        try:
            faiss.write_index(self.index, index_file)
            
            with open(metadata_file, "wb") as f:
                pickle.dump(self.metadata, f)
                
            self.logger.info("Index and metadata saved successfully.")
        except IOError as e:
            self.logger.error("IO error while saving index or metadata: %s", str(e))
            raise IOError("Failed to write index or metadata to disk.") from e
        except Exception as e:
            self.logger.error("Unexpected error during save: %s", str(e))
            raise IOError(f"Unexpected failure while saving index: {e}") from e

    def load_index(self) -> None:
        """
        Loads the FAISS index and metadata from the filesystem.

        Raises:
            FileNotFoundError: If the index or metadata files do not exist.
            IOError: If there is an issue reading the files.
            RuntimeError: If FAISS fails to parse the index file.
        """
        index_file = os.path.join(self.config.vector_db_path, "faiss.index")
        metadata_file = os.path.join(self.config.vector_db_path, "metadata.pkl")

        if not os.path.exists(index_file) or not os.path.exists(metadata_file):
            self.logger.error("Index or metadata file missing at %s", self.config.vector_db_path)
            raise FileNotFoundError("Index or metadata files not found.")

        self.logger.info("Loading FAISS index from %s", index_file)
        try:
            self.index = faiss.read_index(index_file)
            
            with open(metadata_file, "rb") as f:
                self.metadata = pickle.load(f)
                
            self.logger.info("Successfully loaded %d vectors.", self.index.ntotal)
        except IOError as e:
            self.logger.error("IO error while reading index or metadata: %s", str(e))
            raise IOError("Failed to read index or metadata from disk.") from e
        except Exception as e:
            self.logger.error("Failed to parse FAISS index: %s", str(e))
            raise RuntimeError(f"Corrupted or invalid FAISS index file: {e}") from e

    def search(
        self, 
        query_embedding: np.ndarray, 
        top_k: int = 5
    ) -> List[Tuple[float, Dict[str, Optional[str]]]]:
        """
        Retrieves the top_k most similar vectors and their metadata.

        Args:
            query_embedding: A 2D numpy array containing a single query vector 
                             of shape (1, embedding_dim).
            top_k: The number of nearest neighbors to retrieve.

        Returns:
            List[Tuple[float, Dict[str, Optional[str]]]]: A list of tuples, where each tuple
            contains the L2 distance (float) and the corresponding metadata dictionary.

        Raises:
            RuntimeError: If the index is not initialized.
            ValueError: If the query_embedding shape is invalid.
            TypeError: If query_embedding is not a numpy array.
        """
        if self.index is None:
            self.logger.error("Search called on uninitialized index.")
            raise RuntimeError("Index must be built or loaded before searching.")

        if not isinstance(query_embedding, np.ndarray):
            self.logger.error("Query embedding must be a numpy array.")
            raise TypeError("Expected query_embedding to be a numpy array.")

        if query_embedding.ndim != 2 or query_embedding.shape[0] != 1:
            self.logger.error("Invalid query_embedding shape: %s", query_embedding.shape)
            raise ValueError("Query embedding must have shape (1, embedding_dim).")

        # FAISS expects float32
        query_float32 = np.ascontiguousarray(query_embedding, dtype=np.float32)

        self.logger.debug("Executing similarity search for top_k=%d", top_k)
        distances, indices = self.index.search(query_float32, top_k)

        results: List[Tuple[float, Dict[str, Optional[str]]]] = []
        # Flatten the arrays since we only have one query
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1 and idx < len(self.metadata):
                results.append((float(dist), self.metadata[idx]))

        return results