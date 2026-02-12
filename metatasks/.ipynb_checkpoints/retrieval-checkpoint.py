import numpy as np
from typing import Any, Dict, List
from base.base import AbsTask, AbsModel, AbsMethod

class RetrievalTask(AbsTask):
    """Task for retrieval evaluation (e.g., Image-Text retrieval)."""
    
    def __init__(self, name: str, queries: np.ndarray, documents: np.ndarray, gt_ids: np.ndarray, support_embeddings: Dict[str, np.ndarray] = None, topk: int = 5, num_gt: int = 1):
        super().__init__(name, "retrieval")
        self.queries = queries
        self.documents = documents
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
            
            all_hits = []
            for idx in range(aligned_queries.shape[0]):
                gt_query_ids = self.gt_ids[idx * self.num_gt : (idx + 1) * self.num_gt]
                q_emb = aligned_queries[idx, :].reshape(1, -1)
                q_emb_expanded = np.repeat(q_emb, aligned_documents.shape[0], axis=0)
            
            if hasattr(method, 'similarity_function'):
                similarity_function = method.get_similarity_function()
            else:
                def similarity_function(x, y):
                    x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-10)
                    y = y / (np.linalg.norm(y, axis=1, keepdims=True) + 1e-10)
                    return np.sum(x * y, axis=1)
            
            sim_scores = similarity_function(q_emb_expanded, aligned_documents)

            # Get topk indices
            sim_top_idx = np.argpartition(sim_scores, -self.topk)[-self.topk:]
            sim_top_idx = sim_top_idx[np.argsort(sim_scores[sim_top_idx])[::-1]]
            
            hit = np.zeros((self.topk, self.num_gt))
            for jj in range(self.num_gt):
                for ii in range(self.topk):
                    # Here we assume gt_ids matches indices of documents logic from original retrieval file
                     if idx * self.num_gt < len(self.gt_ids):
                        hit[ii, jj] = 1 if gt_query_ids[jj] == self.gt_ids[sim_top_idx[ii]] else 0
            all_hits.append(hit)

        # Calculate metrics
        precisions = []
        for hit in all_hits:
            hit_mean = np.mean(hit, axis=1)
            precision = np.cumsum(hit_mean) / (np.arange(self.topk) + 1)
            precisions.append(precision)
            
        avg_precisions = np.array(precisions).mean(axis=0)
        return {
            "p@1": avg_precisions[0],
            "p@5": avg_precisions[4] if self.topk >= 5 else None
        }
