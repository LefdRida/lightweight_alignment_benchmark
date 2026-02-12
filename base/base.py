from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
import numpy as np

class AbsTask(ABC):
    """Abstract Base Class for all tasks (Classification, Retrieval, etc.)"""
    
    def __init__(self, name: str, task_type: str):
        self.name = name
        self.task_type = task_type
        self.dataset = None


    @abstractmethod
    def run(self, method: 'AbsMethod', model: 'AbsModel', **kwargs) -> Dict[str, Any]:
        """Run the evaluation for this task."""
        pass

class AbsModel(ABC):
    """Abstract Base Class for models (encoders)"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    def encode_image(self, images: Any) -> np.ndarray:
        """Encode images into embeddings."""
        pass

    @abstractmethod
    def encode_text(self, text: List[str]) -> np.ndarray:
        """Encode text into embeddings."""
        pass

class AbsMethod(ABC):
    """Abstract Base Class for alignment methods (ASIF, CSA, etc.)"""
    
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def align(self, query_embeddings: np.ndarray, support_embeddings: Dict[str, np.ndarray], **kwargs) -> np.ndarray:
        """Align representations from one space to another or create a joint space."""
        pass
