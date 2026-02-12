"""Code modified from the original implementation by the authors of the ASIF paper.

Paper title: ASIF: Coupled Data Turns Unimodal Models to Multimodal Without Training (https://openreview.net/pdf?id=YAxV_Krcdjm).
Github: https://colab.research.google.com/github/noranta4/ASIF/blob/main/ASIF_colab_demo.ipynb#scrollTo=x51O0Ndmj1Sy
"""

import torch


def relative_represent(
    y: torch.Tensor, basis: torch.Tensor, non_zeros: int = 800, max_gpu_mem_gb: int = 8
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the sparse decomposition of a tensor y with respect to a basis, considering the available GPU memory.

    Args:
        y (torch.Tensor): Vectors to represent.
        basis (torch.Tensor): Basis to represent with respect to.
        non_zeros (int): Nonzero entries in the relative representation.
        max_gpu_mem_gb (int): Maximum GPU memory allowed to use in gigabytes.

    Returns:
        indices (torch.Tensor): Indices of the nonzero entries in each relative representation of y.
        values (torch.Tensor): Corresponding coefficients of the entries.
    """
    values, indices = torch.zeros((y.shape[0], non_zeros)), torch.zeros(
        (y.shape[0], non_zeros), dtype=torch.long
    )

    free_gpu_mem = max_gpu_mem_gb * 1024**3
    max_floats_in_mem = free_gpu_mem / 4
    max_chunk_y = max_floats_in_mem / basis.shape[0]
    n_chunks = int(y.shape[0] / max_chunk_y) + 1
    chunk_y = int(y.shape[0] / n_chunks) + n_chunks

    with torch.no_grad():
        for c in range(n_chunks):
            in_prods = torch.einsum(
                "ik, jk -> ij", y[c * chunk_y : (c + 1) * chunk_y], basis
            )
            (
                values[c * chunk_y : (c + 1) * chunk_y],
                indices[c * chunk_y : (c + 1) * chunk_y],
            ) = torch.topk(in_prods, non_zeros, dim=1)
            del in_prods

    return indices.to("cpu"), values.to("cpu")


def sparsify(
    i: torch.Tensor, v: torch.Tensor, size: torch.Size
) -> torch.sparse.FloatTensor:
    """Organize indices and values of n vectors into a single sparse tensor.

    Args:
        i (torch.Tensor): indices of non-zero elements of every vector. Shape: (n_vectors, nonzero elements)
        v (torch.Tensor): values of non-zero elements of every vector. Shape: (n_vectors, nonzero elements)
        size (torch.Size): shape of the output tensor

    Returns:
        torch.sparse.FloatTensor: sparse tensor of shape "size" (n_vectors, zero + nonzero elements)
    """
    flat_dim = len(i.flatten())
    coo_first_row_idxs = torch.div(
        torch.arange(flat_dim), i.shape[1], rounding_mode="floor"
    )
    stacked_idxs = torch.cat(
        (coo_first_row_idxs.unsqueeze(0), i.flatten().unsqueeze(0)), 0
    )
    return torch.sparse_coo_tensor(stacked_idxs, v.flatten(), size)


def normalize_sparse(
    tensor: torch.sparse.FloatTensor, nnz_per_row: int
) -> torch.sparse.FloatTensor:
    """Normalize a sparse tensor by row.

    Args:
        tensor (torch.sparse.FloatTensor): The sparse tensor to normalize.
        nnz_per_row (int): The number of non-zero elements per row.

    Returns:
        torch.sparse.FloatTensor: The normalized sparse tensor.
    """
    norms = torch.sparse.sum(tensor * tensor, dim=1).to_dense()
    v = tensor._values().clone().detach().reshape(-1, nnz_per_row).t()  # noqa: SLF001
    v /= torch.sqrt(norms)
    tensor_idx = tensor._indices()  # noqa: SLF001
    return torch.sparse_coo_tensor(tensor_idx, v.t().flatten(), tensor.shape)


