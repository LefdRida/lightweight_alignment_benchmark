from typing import Union, List, Optional
from base.base import AbsTask
from metatasks import get_metatask, list_metatasks
from omegaconf import DictConfig

# Use registry for plug-and-play dataset loading
from data import get_dataset_class, list_datasets


def _load_embeddings_and_split(dataset, split: str):
    """Helper function to load embeddings and optionally create train/test split.
    
    Args:
        dataset: Dataset instance
        split: Dataset split type ('large', 'train', or 'val')
        
    Returns:
        support_embeddings dict or None
    """
    dataset.load_two_encoder_data()
    if dataset.metatask == "classification":
        dataset.set_labels_emb()

    if split == 'large' or split == 'train':
        dataset.set_train_test_split_index()
        dataset.set_training_paired_embeddings()

    support_embeddings = dataset.get_support_embeddings()
    return support_embeddings


def load_dataset_metatask(dataset_name: str, dataset_config: DictConfig) -> AbsTask:
    """Load a dataset and wrap it into a MetaTask.
    
    Args:
        dataset_name: Name of the dataset (imagenet-1k, flickr30k, mscoco-*)
        dataset_config: Configuration dictionary/object for the dataset
        
    Returns:
        Task instance (ClassificationTask or RetrievalTask)
        
    Raises:
        ValueError: if dataset name is not supported
    """
    dataset_name_lower = dataset_name.lower()
    try:
        DatasetClass = get_dataset_class(f"{dataset_name_lower}-{dataset_config.metatask}")
    except ValueError:
        print(f"Dataset {dataset_name} not found. Available: {list_datasets()}")
        exit(1)
    
    ds = DatasetClass(dataset_config)
    support_embeddings = _load_embeddings_and_split(ds, ds.split)
    
    # Validate metatask using registry
    try:
        metatask = get_metatask(dataset_config.metatask)
        return metatask(f"{dataset_name_lower}-{dataset_config.metatask}", ds, dataset_config, support_embeddings)

    except ValueError:
        raise ValueError(
            f"Metatask '{dataset_config.metatask}' not supported. "
            f"Available: {list_metatasks()}"
        )


