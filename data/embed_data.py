from abc import abstractmethod
from collections.abc import Callable
import numpy as np
from typing import Tuple

def embed_images(
    image_paths: list,
    image_encoder: callable,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """Embed the dataset images using the provided encoder.

    Args:
        image_encoder: callable that takes a list of image paths and returns their embeddings

    Returns:
        image_emb: numpy array of image embeddings
    """
    return image_encoder(image_paths, **kwargs)

def embed_text(
    text: list,
    text_encoder: callable,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """Embed the dataset texts using the provided encoder.

    Args:
        text_encoder: callable that takes a list of text descriptions and returns their embeddings

    Returns:
        text_emb: numpy array of text embeddings
    """
    return text_encoder(text, **kwargs)