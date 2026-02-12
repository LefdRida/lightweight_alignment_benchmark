import torch
import numpy as np
from typing import Any, List, Union
from base.base import AbsModel

class HFModelWrapper(AbsModel):
    """Wrapper for HuggingFace models."""
    
    def __init__(self, model_id: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__(model_id)
        from transformers import AutoModel, AutoTokenizer, AutoImageProcessor
        self.device = device
        self.model = AutoModel.from_pretrained(model_id).to(device)
        self.model.eval()
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        except:
            self.tokenizer = None
            
        try:
            self.image_processor = AutoImageProcessor.from_pretrained(model_id)
        except:
            self.image_processor = None

    def encode_image(self, images: List[Any]) -> np.ndarray:
        if self.image_processor is None:
            raise ValueError("No image processor found for this model.")
        
        inputs = self.image_processor(images=images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.get_image_features(**inputs) if hasattr(self.model, "get_image_features") else self.model(**inputs).last_hidden_state[:, 0]
        return outputs.cpu().numpy()

    def encode_text(self, text: List[str]) -> np.ndarray:
        if self.tokenizer is None:
            raise ValueError("No tokenizer found for this model.")
        
        inputs = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.get_text_features(**inputs) if hasattr(self.model, "get_text_features") else self.model(**inputs).last_hidden_state[:, 0]
        return outputs.cpu().numpy()
