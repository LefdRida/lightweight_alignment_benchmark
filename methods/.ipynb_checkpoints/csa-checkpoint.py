import numpy as np
from typing import Dict, Any, Optional
from base.base import AbsMethod
# Importing from original structure
from methods.csa_core import NormalizedCCA

class CSAMethod(AbsMethod):
    """CSA (CCA-based) alignment technique."""
    
    def __init__(self, sim_dim: int = 512):
        super().__init__("CSA")
        self.sim_dim = sim_dim
        self.cca = NormalizedCCA(sim_dim=sim_dim)
        self.fitted = False

    def align(self, image_embeddings: np.ndarray, text_embeddings: np.ndarray, support_embeddings: Dict[str, np.ndarray], **kwargs) -> np.ndarray:
        """
        Align embeddings using CSA.
        """
        if not self.fitted:
            # support_embeddings should have 'train_image' and 'train_text'
            self.cca.fit_transform_train_data(
                support_embeddings['train_image'], 
                support_embeddings['train_text']
            )
            self.fitted = True

        # Transform based on mode
        # Zero mean using training mean
        
        image_embeddings_centred = image_embeddings - self.cca.traindata1_mean
        text_embeddings_centred = text_embeddings - self.cca.traindata2_mean
        print(image_embeddings_centred.shape)
        print(text_embeddings_centred.shape)
        # Transform using CCA weights for first modality
        image_embeddings, text_embeddings = self.cca.transform_data(image_embeddings_centred, text_embeddings_centred)

        return image_embeddings, text_embeddings
    
    def similarity_function(
        self,
        x: np.ndarray,
        y: np.ndarray,
    ) -> np.ndarray:
        """Compute the weighted correlation similarity.

        Args:
            x: data 1. shape: (N, feats)
            y: data 2. shape: (N, feats)
            corr: correlation matrix. shape: (feats, )
            dim: number of dimensions to select

        Return:
            similarity matrix between x and y. shape: (N, )
        """
        assert (
            x.shape == y.shape
        ), f"x and y should have the same number of shape, but got {x.shape} and {y.shape}"
        corr: np.ndarray = self.cca.corr_coeff
        dim: int = self.cca.sim_dim
        # select the first dim dimensions
        x, y, corr = x[:, :dim], y[:, :dim], corr[:dim]
        # normalize x and y with L2 norm
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
        y = y / np.linalg.norm(y, axis=1, keepdims=True)
        # compute the similarity scores
        sim = np.zeros(x.shape[0])
        for ii in range(x.shape[0]):
            sim[ii] = corr * x[ii] @ y[ii]
        return sim
    
    def get_similarity_function(self):
        return self.similarity_function
        
