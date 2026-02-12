import numpy as np
from sklearn import metrics
from typing import Any, Dict, List
from base.base import AbsTask, AbsModel, AbsMethod

class ClassificationTask(AbsTask):
    """Task for zero-shot classification evaluation."""
    
    def __init__(self, name: str, test_images: np.ndarray, support_embeddings: Dict[str, np.ndarray], ground_truth: np.ndarray):
        super().__init__(name, "classification")
        self.test_images = test_images         # Images embeddings or raw images
        self.support_embeddings = support_embeddings       # Support images embeddings
        self.ground_truth = ground_truth   # Ground truth labels

    def run(self, method: AbsMethod, support_embeddings: Dict[str, np.ndarray] = None, **kwargs) -> Dict[str, Any]:
        """Run classification using the provided alignment method."""
        
        if support_embeddings is None:
            support_embeddings = self.support_embeddings
            
        # Generic flow for alignment-based classification:
        if hasattr(method, 'classify'):
             predictions = method.classify(self.test_images, self.support_embeddings.get("labels_emb", None), support_embeddings)
        else:
            # Fallback: align then compare
            aligned_image_embeddings, aligned_labels = method.align(
                image_embeddings=self.test_images,
                text_embeddings=self.support_embeddings.get("labels_emb", None),
                support_embeddings=support_embeddings,
            )
            
            if hasattr(method, 'similarity_function'):
                similarity_function = method.get_similarity_function()
            else:
                def similarity_function(x, y):
                    x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-10)
                    y = y / (np.linalg.norm(y, axis=1, keepdims=True) + 1e-10)
                    return np.sum(x * y, axis=1)
            sim_scores = []
            for label_idx in range(aligned_labels.shape[0]):
                l_emb = aligned_labels[label_idx].reshape(1, -1)
                l_emb = np.repeat(l_emb, aligned_image_embeddings.shape[0], axis=0)
                sim_score_matrix = similarity_function(aligned_image_embeddings, l_emb)
                sim_scores.append(sim_score_matrix)
            
            sim_scores = np.array(sim_scores)
            predictions = np.argmax(sim_scores, axis=0)
            
        if hasattr(method, 'evaluate'):
            results = method.evaluate(self.ground_truth, predictions)
        else:
            accuracy = metrics.accuracy_score(self.ground_truth, predictions)
            results = {"accuracy": accuracy}
        return results
    
    
