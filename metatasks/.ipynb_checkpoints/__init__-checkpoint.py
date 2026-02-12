"""Metatasks Registry for plug-and-play task selection."""
from typing import Type, Dict, Optional
from base.base import AbsTask
from .classification import ClassificationTask
from .retrieval import RetrievalTask
from omegaconf import DictConfig

def _create_classification_task(
    dataset_name: str,
    dataset,
    metatask_config: DictConfig,
    support_embeddings: Optional[dict]
) -> ClassificationTask:
    """Helper function to create a classification task.
    
    Args:
        dataset_name: Name of the dataset
        dataset: Dataset instance with classification data
        support_embeddings: Optional training embeddings
        
    Returns:
        ClassificationTask instance
    """


    test_img, test_labels = dataset.get_test_data()
    
    return ClassificationTask(
        name=f"{dataset_name}-Classification",
        test_images=test_img,
        support_embeddings = support_embeddings,
        ground_truth=test_labels,
    )


def _create_retrieval_task(
    dataset_name: str,
    dataset,
    metatask_config: DictConfig,
    support_embeddings: Optional[dict]
) -> RetrievalTask:
    """Helper function to create a retrieval task.
    
    Args:
        dataset_name: Name of the dataset
        dataset: Dataset instance with retrieval data
        metatask_config: Configuration object
        support_embeddings: Optional training embeddings
        
    Returns:
        RetrievalTask instance
    """
    val_img, val_txt, gt_caption_ids, gt_img_ids = dataset.get_test_data()
    
    return RetrievalTask(
        name=f"{dataset_name}-Retrieval",
        queries=val_img,
        documents=val_txt,
        gt_ids=gt_img_ids,
        support_embeddings=support_embeddings,
        topk=metatask_config.topk,
        num_gt=metatask_config.num_gt
    )

_METATASK_REGISTRY: Dict[str, Type[AbsTask]] = {
    "classification": _create_classification_task,
    "retrieval": _create_retrieval_task,
}


def get_metatask(name: str) -> Type[AbsTask]:
    """
    Retrieve a metatask class by name.
    
    Args:
        name: The name of the metatask to retrieve (e.g., "classification", "retrieval").
        
    Returns:
        The metatask class.
        
    Raises:
        ValueError: If the metatask name is not found in the registry.
    """
    task = _METATASK_REGISTRY.get(name.lower())
    if task is None:
        available = list(_METATASK_REGISTRY.keys())
        raise ValueError(f"Metatask '{name}' not found. Available metatasks: {available}")
    return task


def list_metatasks() -> list[str]:
    """List all available registered metatasks."""
    return list(_METATASK_REGISTRY.keys())
