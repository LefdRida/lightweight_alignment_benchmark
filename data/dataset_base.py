from abc import abstractmethod
import numpy as np
from typing import Tuple, Any, Dict
from .dataset_utils import load_embeddings_from_hf

class DatasetBase:
    """Base class for all datasets."""
    def __init__(self):
        pass

    @abstractmethod
    def load_data(self, *args, **kwargs):
        """Load raw dataset (paths, labels, captions)."""
        pass

    def get_test_data(self) -> Any:
        """Return test data. Override in subclass."""
        raise NotImplementedError("Subclass must implement get_test_data()")

    def get_image_paths(self) -> Any:
        """Return image paths. Override in subclass."""
        return self.image_paths

    def get_captions(self) -> Any:
        """Return text. Override in subclass."""
        return self.captions

    def get_labels(self) -> Any:
        """Return labels. Override in subclass."""
        return self.labels_descriptions

class EmbeddingDataset:
    """Dataset class to hold pre-computed embeddings.
    
    Inherits from DatasetBase to provide a unified interface.
    """
    def __init__(self, 
            split: str = "large"
            ):
        self.split = split

        self.support_embeddings = {}
        self.image_embeddings = None
        self.text_embeddings = None
        self.labels = None
        self.train_idx = None
        self.val_idx = None
        
    def load_two_encoder_data(self, hf_repo_id: str, hf_img_embedding_name: str, hf_text_embedding_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load embeddings for two modalities from HuggingFace Hub.

        Returns:
            Tuple of (image_embeddings, text_embeddings)
        """
        self.image_embeddings = load_embeddings_from_hf(
            hf_file_name=hf_img_embedding_name, 
            repo_id=hf_repo_id
        )
        self.text_embeddings = load_embeddings_from_hf(
            hf_file_name=hf_text_embedding_name, 
            repo_id=hf_repo_id
        )
        return self.image_embeddings, self.text_embeddings

    def set_train_test_split_index(self, train_test_ratio: float = 0.7, seed: int = 42) -> None:
        """Get the index of the training and validation set."""
        assert self.image_embeddings is not None and self.text_embeddings is not None, \
            "Please load the data first."
        assert self.split == "large", "Split must be 'large' to create train/test split."
        
        n = self.image_embeddings.shape[0]
        arange = np.arange(n)
        np.random.seed(seed)
        np.random.shuffle(arange)
        self.train_idx = arange[:int(n * train_test_ratio)]
        self.val_idx = arange[int(n * train_test_ratio):]


    def set_training_paired_embeddings(self) -> None:
        """Set paired embeddings for training."""
        raise NotImplementedError("Subclass must implement set_training_paired_embeddings()")
    
    def get_test_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get test data."""
        raise NotImplementedError("Subclass must implement get_test_data()")

    def get_support_embeddings(self) -> Dict[str, np.ndarray]:
        """Get support embeddings."""
        assert self.support_embeddings is not None, "Please load the data first."
        return self.support_embeddings