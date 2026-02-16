import numpy as np
from typing import Any, Dict, List
from base.base import AbsTask, AbsModel, AbsMethod
import torch 
from tqdm import tqdm
class RetrievalTask(AbsTask):
    """Task for retrieval evaluation (e.g., Image-Text retrieval)."""
    
    def __init__(self, name: str, queries: np.ndarray, documents: np.ndarray, gt_ids: np.ndarray, support_embeddings: Dict[str, np.ndarray] = None, topk: int = 5, num_gt: int = 1):
        super().__init__(name, "retrieval")
        self.queries = np.array(queries)
        self.documents = np.array(documents)
        self.gt_ids = gt_ids
        self.support_embeddings = support_embeddings
        self.topk = topk
        self.num_gt = num_gt


    def run(self, method: AbsMethod, support_embeddings: Dict[str, np.ndarray] = None, **kwargs) -> Dict[str, Any]:
        """Run retrieval using the provided alignment method."""
        
        if support_embeddings is None:
            support_embeddings = self.support_embeddings
        # Align queries and/or documents
        if hasattr(method, 'retrieve'):
            all_hits = method.retrieve(self.queries, self.gt_ids, self.documents, support_embeddings, self.topk, self.num_gt)
        else:
            aligned_queries, aligned_documents = method.align(
                image_embeddings=self.queries,
                text_embeddings=self.documents,
                support_embeddings=support_embeddings
                )

            if hasattr(method, 'similarity_function'):
                similarity_function = method.get_similarity_function()
            else:
                def similarity_function(x, y):
                    x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-10)
                    y = y / (np.linalg.norm(y, axis=1, keepdims=True) + 1e-10)
                    return np.sum(x * y, axis=1)
            
            all_hits = []
            
            for idx in tqdm(range(aligned_queries.shape[0])):
                gt_query_ids = self.gt_ids[idx]
                q_emb = aligned_queries[idx, :].reshape(1, -1)
                sim_scores = similarity_function(q_emb, aligned_documents)
                # Get topk indices
                sim_top_idx = torch.topk(sim_scores, self.topk, largest=True, sorted=True).indices.cpu().numpy()
                hit = np.zeros(self.topk)
                for jj, top_idx in enumerate(sim_top_idx):
                    hit[jj] = 1 if top_idx in gt_query_ids else 0
                all_hits.append(hit)
            

        # Calculate metrics
        precisions = []
        recalls = []
        for hit in all_hits:
            precision = np.cumsum(hit) / (np.arange(self.topk) + 1)
            precisions.append(precision)
            # TODO: check this
            recall = np.cumsum(hit) / self.num_gt
            recalls.append(recall)
            
        avg_precisions = np.array(precisions).mean(axis=0)
        avg_recalls = np.array(recalls).mean(axis=0)
        return {
            "p@1": avg_precisions[0],
            "p@5": avg_precisions[4] if self.topk >= 5 else None,
            "r@1": avg_recalls[0],
            "r@5": avg_recalls[4] if self.topk >= 5 else None,
        }
