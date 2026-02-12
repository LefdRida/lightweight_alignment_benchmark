import torch
import numpy as np
from typing import Dict, Any, Optional
from base.base import AbsMethod
# Importing from local module
from methods.asif_core import relative_represent, sparsify, normalize_sparse

class ASIFMethod(AbsMethod):
    """ASIF alignment technique."""
    
    def __init__(self, non_zeros: int = 800, val_exps: list = [1.0], max_gpu_mem_gb: float = 8.0):
        super().__init__("ASIF")
        self.non_zeros = non_zeros
        self.val_exps = val_exps
        self.max_gpu_mem_gb = max_gpu_mem_gb

    def align(self):
        pass
    
    def retrieve(
        self,
        queries: np.ndarray,
        gt_document_ids: np.ndarray,
        documents: np.ndarray,
        support_embeddings: Dict[str, np.ndarray],
        topk: int = 5,
        num_gt: int = 1,
        **kwargs
    ) -> np.ndarray:
        assert topk >= num_gt, "topk should be more than num_gt"
        assert gt_document_ids.shape[0] == documents.shape[0], "gt_document_ids and documents should have the same number of samples"
        
        non_zeros = min(self.non_zeros, support_embeddings['train_image'].shape[0])
        range_anch = [
            2**i
            for i in range(
                int(np.log2(non_zeros) + 1),
                int(np.log2(len(support_embeddings['train_image']))) + 2,
            )
        ]
        range_anch = range_anch[-1:]  # run just last anchor to be quick
        val_labels = torch.zeros((1,), dtype=torch.float32)
        
        _, _, sim_score_matrix = self.similarity_function(
            torch.tensor(queries, dtype=torch.float32),
            torch.tensor(documents, dtype=torch.float32),
            torch.tensor(support_embeddings['train_image'], dtype=torch.float32),
            torch.tensor(support_embeddings['train_text'], dtype=torch.float32),
            val_labels,
            non_zeros,
            range_anch,
            self.val_exps,
            max_gpu_mem_gb=self.max_gpu_mem_gb,
        )
        sim_score_matrix = sim_score_matrix.numpy().astype(np.float32)

        
        self.sim_scores = []
        self.all_hit = []
        for idx in range(0, queries.shape[0]):
            gt_query_ids = gt_document_ids[idx*num_gt:(idx+1)*num_gt]
            # copy the test text to the number of images
            sim_score = sim_score_matrix[idx, :]

            # sort the similarity score in descending order and get the index
            sim_top_idx = np.argpartition(sim_score, -num_gt)[-num_gt :]
            sim_top_idx = sim_top_idx[np.argsort(sim_score[sim_top_idx])[::-1]]
            hit = np.zeros((topk, num_gt))
            for jj in range(num_gt):
                for ii in range(topk):
                    hit[ii, jj] = 1 if gt_query_ids[jj] == gt_document_ids[sim_top_idx[ii]] else 0
            self.all_hit.append(hit)
            self.sim_scores.append(sim_score)
        return self.all_hit
    
    def classify(
        self, 
        data: np.ndarray, 
        labels_emb: np.ndarray,
        support_embeddings: Dict[str, np.ndarray]
        ) -> np.ndarray:

        non_zeros = min(self.non_zeros, support_embeddings["train_image"].shape[0])
        range_anch = [
            2**i
            for i in range(
                int(np.log2(self.non_zeros) + 1),
                int(np.log2(len(support_embeddings["train_image"])) + 2),
            )
        ]
        range_anch = range_anch[-1:]  # run just last anchor to be quick
        val_labels = torch.zeros((1,), dtype=torch.float32)
        # generate noise in the shape of the labels_emb
        noise = np.random.rand(
            data.shape[0] - labels_emb.shape[0],
            labels_emb.shape[1],
        ).astype(np.float32)
        test_label = np.concatenate((labels_emb, noise), axis=0)
        assert (
            data.shape[0] == test_label.shape[0]
        ), f"{data.shape[0]}!={test_label.shape[0]}"
        _, _, sim_score_matrix = self.similarity_function(
            torch.tensor(data, dtype=torch.float32),
            torch.tensor(test_label, dtype=torch.float32),
            torch.tensor(support_embeddings["train_image"], dtype=torch.float32),
            torch.tensor(support_embeddings["train_text"], dtype=torch.float32),
            val_labels,
            non_zeros,
            range_anch,
            self.val_exps,
            max_gpu_mem_gb=self.max_gpu_mem_gb,
        )
        sim_score_matrix = sim_score_matrix.numpy().astype(np.float32)[:, :2]
        sim_scores = sim_score_matrix.T
        predictions = np.argmax(sim_scores, axis=0)
        return predictions
    
    def similarity_function(
        self,
        zimgs: torch.Tensor,
        ztxts: torch.Tensor,
        aimgs: torch.Tensor,
        atxts: torch.Tensor,
        test_labels: list,
        non_zeros: int,
        range_anch: range,
        val_exps: list,
        dic_size: int = 100_000,
        max_gpu_mem_gb: float = 8.0,
    ) -> tuple[list, dict, torch.Tensor]:
        """Computes the zero-shot classification accuracy using relative representations over sets of anchors.

        Args:
            zimgs (torch.Tensor): absolute embeddings of the images
            ztxts (torch.Tensor): absolute embeddings of the texts
            aimgs (torch.Tensor): absolute embeddings of the anchor images
            atxts (torch.Tensor): absolute embeddings of the anchor texts
            test_labels (list): ground truth labels of the images
            non_zeros (int): nonzero entries in the relative representation
            range_anch (range): range of sizes of the anchor's sets to use (overshoot is ok)
            val_exps (list): similarity exponents to test
            dic_size (int): size of the chunk of aimgs to load in memory to fit all intermediate variables in RAM
            max_gpu_mem_gb (float): maximum GPU memory allowed to use in gigabytes
        Returns:
            n_anchors (list): list of sizes of the anchor's sets (with overshooting fixed)
            scores (dict): dictionary of scores for each tested similarity exponent
            sims (torch.Tensor): similarity matrix between images and texts
        """
        n_anchors = []
        scores = {ve: [] for ve in val_exps}
        n_templates = max(
            int(ztxts.shape[0] / (max(test_labels) - min(test_labels) + 1)), 1
        )

        for i in range_anch:
            sims = torch.zeros((len(zimgs), len(ztxts)))
            idxs_imgs = torch.zeros(((len(zimgs), non_zeros * 2)), dtype=torch.long)
            idxs_txts = torch.zeros(((len(ztxts), non_zeros * 2)), dtype=torch.long)
            vals_imgs = torch.zeros((len(zimgs), non_zeros * 2))
            vals_txts = torch.zeros((len(ztxts), non_zeros * 2))

            for d in range(min(len(aimgs), i) // (dic_size + 1) + 1):
                idxs, values = relative_represent(
                    zimgs,
                    aimgs[d * dic_size : min(i, (d + 1) * dic_size)],
                    non_zeros=non_zeros,
                    max_gpu_mem_gb=max_gpu_mem_gb,
                )
                idxs_imgs[:, :non_zeros] = idxs + d * dic_size
                vals_imgs[:, :non_zeros] = values
                idxs, values = relative_represent(
                    ztxts,
                    atxts[d * dic_size : min(i, (d + 1) * dic_size)],
                    non_zeros=non_zeros,
                    max_gpu_mem_gb=max_gpu_mem_gb,
                )
                idxs_txts[:, :non_zeros] = idxs + d * dic_size
                vals_txts[:, :non_zeros] = values

                top_valsi, indices = torch.topk(vals_imgs, non_zeros, dim=1)
                top_idxsi = torch.gather(idxs_imgs, 1, indices)
                top_valst, indices = torch.topk(vals_txts, non_zeros, dim=1)
                top_idxst = torch.gather(idxs_txts, 1, indices)

                idxs_imgs[:, non_zeros:] = top_idxsi
                vals_imgs[:, non_zeros:] = top_valsi
                idxs_txts[:, non_zeros:] = top_idxst
                vals_txts[:, non_zeros:] = top_valst

            for val_exp in val_exps:
                ztxts_t = sparsify(
                    top_idxst, top_valst**val_exp, (len(ztxts), min(len(aimgs), i))
                ).to(zimgs.device)
                ztxts_t = normalize_sparse(ztxts_t, non_zeros)

                if (
                    i < max_gpu_mem_gb * 1024**3 / 4 / zimgs.shape[0]
                ):  # einsum until it fits in GPU memory
                    zimgs_t = sparsify(
                        top_idxsi, top_valsi**val_exp, (len(zimgs), min(len(aimgs), i))
                    ).to(zimgs.device)
                    sims = torch.einsum(
                        "ij, kj -> ik", zimgs_t.to_dense(), ztxts_t.to_dense()
                    ).to("cpu")
                else:
                    n_chunks = 6
                    zs = zimgs.shape[0]
                    chunks = [c * (zs // n_chunks) for c in range(n_chunks)] + [zs]
                    for ci in range(n_chunks):
                        zimgs_t = sparsify(
                            top_idxsi[chunks[ci] : chunks[ci + 1]],
                            top_valsi[chunks[ci] : chunks[ci + 1]] ** val_exp,
                            (chunks[ci + 1] - chunks[ci], min(len(aimgs), i)),
                        ).to(zimgs.device)
                        sims[chunks[ci] : chunks[ci + 1]] = (
                            torch.sparse.mm(zimgs_t, ztxts_t.t()).to("cpu").to_dense()
                        )
                score = float(
                    (
                        torch.div(sims.argmax(axis=1), n_templates, rounding_mode="floor")
                        == test_labels.clone().detach()
                    ).sum()
                    / len(zimgs)
                )
                scores[val_exp].append(score)
            n_anchors.append(min(len(aimgs), i))
        return n_anchors, scores, sims
    
