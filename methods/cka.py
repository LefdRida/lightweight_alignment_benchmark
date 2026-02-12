import numpy as np
from sklearn import datasets, cluster
import torch
import os
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import pandas as pd
from .cka_core import * 



import numpy as np
from typing import Dict, Any, Optional
from base.base import AbsMethod
# Importing from original structure

class CKAMethod(AbsMethod):
    """CSA (CCA-based) alignment technique."""
    
    def __init__(self, method: str = "cka"):
        super().__init__("CKA")
        self.method = method
        self.stretch = False
        self.fitted = False
        self.seed = 0
        self.seed2 = 0
        self.base_samples = 100
        self.query_samples = 100
        self.clustering_mode = 0
        self.same = False
        self.graph_func = linear_local_CKA
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def align(
        self, 
        image_embeddings: np.ndarray, 
        text_embeddings: np.ndarray, 
        support_embeddings: Dict[str, np.ndarray], 
        **kwargs
        ) -> np.ndarray:
        """
        Align embeddings using CSA.
        """

        (source_base, 
        target_base, 
        source_query, 
        target_query ) = get_data_sep(
                0, 
                self.seed, 
                self.base_samples, 
                self.query_samples,  
                support_embeddings['train_image'], 
                support_embeddings['train_text'], 
                image_embeddings, 
                text_embeddings,    
                kwargs['source_base_cluster'],
                kwargs['target_base_cluster'],
                self.clustering_mode, 
                self.same,
                self.stretch
            )
        
        self.graph = self.graph_func(source_base, target_base, source_query, target_query, self.device)
        
    
    def retrieve(
        self, 
        image_embeddings: np.ndarray, 
        text_embeddings: np.ndarray, 
        support_embeddings: Dict[str, np.ndarray], 
        topk: int = 5,
        num_gt: int = 1,
        **kwargs
        ) -> np.ndarray:

        self.align(
            image_embeddings, 
            text_embeddings, 
            support_embeddings, 
            **kwargs
        )

        if self.graph is None:
            return None

        top_doc = 0
        total = 0

        for i in tqdm(range(len(self.graph))):
            row = self.graph[i]
            ind_row = sorted(list(range(len(self.graph))), key = lambda x: -row[x])
        
            if i in ind_row[:topk]:
                top_doc += 1

            total += 1
        
        return top_doc / total

    
    def classify(
        self, 
        image_embeddings: np.ndarray, 
        text_embeddings: np.ndarray, 
        support_embeddings: Dict[str, np.ndarray], 
        **kwargs
        ):
        self.align(
            image_embeddings, 
            text_embeddings, 
            support_embeddings, 
            **kwargs
        )
        torch.manual_seed(0)
        shuffle = torch.randperm(self.graph.shape[1])
        
        graph_shuffled = self.graph[:, shuffle]
        row_ind, col_ind = linear_sum_assignment(graph_shuffled, maximize=True)
        return sum(col_ind[shuffle] == row_ind)/len(col_ind)
 