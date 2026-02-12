# Defer loader import to avoid circular dependency
# from .loader import load_dataset_task  # -> moved to bottom

from typing import Type, Dict
import logging

from .dataset_base import DatasetBase

# Import dataset classes to register them
# Using absolute imports to ensure reliability
from datasets.imagenet1k.imagenet1k_zeroshot_classif_dataset import (
    Imagenet1kZeroshotClassificationDataset, 
)
from datasets.flickr30k.flickr30k_retrieval_dataset import (
    Flickr30kRetrievalDataset, 
)
from datasets.mscoco.mscoco_multilabel_classification_dataset import (
    MScocoMultiLabelClassificationDataset, 
    MScocoRetrievalDataset
)

logger = logging.getLogger(__name__)

# Registry dictionary mapping names to classes
# Registry dictionary mapping names to classes
_DATASET_REGISTRY: Dict[str, Type[DatasetBase]] = {
    # Classification Datasets
    "imagenet1k-classification": Imagenet1kZeroshotClassificationDataset,
    "mscoco-classification": MScocoMultiLabelClassificationDataset,
    
    # Retrieval Datasets
    "flickr30k-retrieval": Flickr30kRetrievalDataset,
    "mscoco-retrieval": MScocoRetrievalDataset,

    # Embedding Generation Datasets (mapped to same classes, logic inside class handles generation mode)
    "imagenet1k-classification-embedding": Imagenet1kZeroshotClassificationDataset,
    "flickr30k-retrieval-embedding": Flickr30kRetrievalDataset,
    "mscoco-retrieval-embedding": MScocoRetrievalDataset,
    "mscoco-classification-embedding": MScocoMultiLabelClassificationDataset,
}

def get_dataset_class(name: str) -> Type[DatasetBase]:
    """
    Retrieve a dataset class by name.
    
    Args:
        name: The name of the dataset to retrieve.
        
    Returns:
        The dataset class.
        
    Raises:
        ValueError: If the dataset name is not found in the registry.
    """
    dataset_class = _DATASET_REGISTRY.get(name.lower())
    if dataset_class is None:
        available = list(_DATASET_REGISTRY.keys())
        raise ValueError(f"Dataset '{name}' not found. Available datasets: {available}")
    return dataset_class

def list_datasets() -> list[str]:
    """List all available registered datasets."""
    return list(_DATASET_REGISTRY.keys())

# Deferred import to prevent circular dependency
from .loader import load_dataset_metatask