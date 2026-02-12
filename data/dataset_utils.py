"""This module contains utility functions for loading data feature embeddings."""

from huggingface_hub import hf_hub_download, HfApi
import pickle
import numpy as np
from pathlib import Path
from typing import Optional
import logging

# Setup logging
logger = logging.getLogger(__name__)

# Simple in-memory cache for embeddings
_EMBEDDING_CACHE = {}


def load_embeddings_from_hf(
        hf_file_name: str,
        repo_id: str,
        use_cache: bool = True
    ) -> np.ndarray:
    """Load embeddings from Hugging Face Hub with caching support.
    
    Args:
        hf_file_name: name of the file in Hugging Face Hub
        repo_id: repository id in Hugging Face Hub
        use_cache: whether to use in-memory cache for loaded embeddings
        
    Returns:                                             
        emb: loaded embeddings
        
    Raises:
        FileNotFoundError: if the embedding file cannot be found
        ValueError: if the loaded data is not a valid numpy array
    """
    cache_key = f"{repo_id}/{hf_file_name}"
    
    # Check cache first
    if use_cache and cache_key in _EMBEDDING_CACHE:
        logger.info(f"Loading embeddings from cache: {cache_key}")
        return _EMBEDDING_CACHE[cache_key]
    
    try:
        logger.info(f"Downloading embeddings from HF: {cache_key}")
        embed_fpath = hf_hub_download(repo_id=repo_id, filename=hf_file_name)
        
        with open(embed_fpath, 'rb') as file:
            emb = pickle.load(file)
        
        # Validate the loaded embeddings
        if not isinstance(emb, np.ndarray):
            raise ValueError(f"Expected numpy array, got {type(emb)}")
        
        logger.info(f"Loaded embeddings shape: {emb.shape}")
        
        # Cache the embeddings
        if use_cache:
            _EMBEDDING_CACHE[cache_key] = emb
            
        return emb
        
    except Exception as e:
        logger.error(f"Failed to load embeddings from {cache_key}: {e}")
        raise


def validate_embeddings(
        embeddings: np.ndarray,
        expected_dims: Optional[int] = None,
        min_samples: int = 1
    ) -> bool:
    """Validate embedding array.
    
    Args:
        embeddings: numpy array to validate
        expected_dims: expected embedding dimension (optional)
        min_samples: minimum number of samples required
        
    Returns:
        True if valid
        
    Raises:
        ValueError: if validation fails
    """
    if not isinstance(embeddings, np.ndarray):
        raise ValueError(f"Embeddings must be numpy array, got {type(embeddings)}")
    
    if embeddings.ndim != 2:
        raise ValueError(f"Embeddings must be 2D array, got shape {embeddings.shape}")
    
    if embeddings.shape[0] < min_samples:
        raise ValueError(f"Need at least {min_samples} samples, got {embeddings.shape[0]}")
    
    if expected_dims is not None and embeddings.shape[1] != expected_dims:
        raise ValueError(f"Expected {expected_dims} dimensions, got {embeddings.shape[1]}")
    
    return True


def clear_embedding_cache():
    """Clear the in-memory embedding cache."""
    global _EMBEDDING_CACHE
    _EMBEDDING_CACHE.clear()
    logger.info("Embedding cache cleared")


def upload_embeddings_to_hf(
        embeddings: np.ndarray, 
        embeddings_saving_path: str, 
        hf_api: HfApi, 
        repo_id: str, 
        path_in_repo: str
    ) -> None:
    """Upload embeddings to Hugging Face Hub.
    
    Args:
        embeddings: numpy array of embeddings to upload
        embeddings_saving_path: path to save embeddings locally before upload
        hf_api: Hugging Face API instance
        repo_id: repository id in Hugging Face Hub
        path_in_repo: path within the repository
        
    Raises:
        ValueError: if embeddings are invalid
    """
    # Validate before uploading
    validate_embeddings(embeddings)
    
    # Ensure parent directory exists
    save_path = Path(embeddings_saving_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(embeddings_saving_path, "wb") as f:
            pickle.dump(embeddings, f)
        
        logger.info(f"Uploading embeddings to {repo_id}/{path_in_repo}")
        hf_api.upload_file(
            path_or_fileobj=embeddings_saving_path,    
            path_in_repo=path_in_repo,   
            repo_id=repo_id,
            repo_type="model",                     
        )
        logger.info("Upload successful")
        
    except Exception as e:
        logger.error(f"Failed to upload embeddings: {e}")
        raise

