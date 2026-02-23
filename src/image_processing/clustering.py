"""
Clustering utilities for character images.

Provides GMM-based initial clustering and tree-based refinement.
"""

from __future__ import annotations

import numpy as np
import torch
from joblib import delayed, Parallel
from scipy.stats import norm
from sklearn.decomposition import PCA
from tqdm import tqdm

from .tools.gmm import train_GMM
from .tools.inverse_compositional import register2mean
from .tools.normality import TESTS


def clusterize_gmm(
    imgs: np.ndarray,
    pca_level: float = 0.9,
    n_comps: int = 70,
    seed: int = 42,
    device: str = "cpu",
) -> np.ndarray:
    """Clusterize images using Gaussian Mixture Models.

    Args:
        imgs: Batch of images, shape (n, h, w).
        pca_level: Ratio of variance to keep with PCA.
        n_comps: Number of components for GMM.
        seed: Random seed for GMM.
        device: PyTorch device on which to perform computations.

    Returns:
        Cluster assignment for each sample, int array of shape (n,).
    """
    # Reduce dimensionality with PCA: keep pca_level ratio of variance
    X = imgs.reshape(len(imgs), -1)
    svd_solver = "covariance_eigh" if X.shape[1] < 1000 else "full"
    Y = PCA(n_components=pca_level, svd_solver=svd_solver).fit_transform(X)

    # Apply EM
    Z = torch.tensor(Y, dtype=torch.float32, device=device)
    _, _, _, idx, _, _ = train_GMM(Z, n_comps, max_steps=200, seed=seed, verbose=True)

    # Cluster assignment for each sample
    clu_pred = np.argmax(idx.cpu().numpy(), axis=0)
    return clu_pred


def grow_tree(
    imgs_root: np.ndarray,
    transform: str,
    min_imgs: int,
    pca_level: float | int,
    num_tests: int,
    test: callable,
    pval_thr: float,
    seed: int,
    device: str,
    max_depth: int = 10,
    skip_registration: bool = False,
) -> list[np.ndarray]:
    """Grow a tree of subclusters from a root cluster.

    Args:
        imgs_root: Images of the root cluster, shape (n, h, w).
        transform: Name of the type of transformations for registration.
        min_imgs: Minimum number of images in a cluster to split.
        pca_level: If int, number of PCA components; if float, variance ratio.
        num_tests: Number of PCA components for normality testing.
        test: Normality test function; takes 1D sample, returns p-value.
        pval_thr: Threshold for normality test p-value.
        seed: Random seed for GMM.
        device: PyTorch device for computations.
        max_depth: Maximum depth of the tree.
        skip_registration: If True, skip expensive image registration.

    Returns:
        List of valid leaf clusters (>= min_imgs elements), each a 1D int
        array of indexes in imgs_root.
    """
    subclusters = []
    edges = []

    pile = [(None, np.arange(len(imgs_root)), 0)]  # (parent_idx, idxs_clu, depth)

    while len(pile) > 0:
        parent_idx, idxs_clu, depth = pile.pop()

        clu_idx = len(subclusters)
        subclusters.append(idxs_clu)
        if parent_idx is not None:
            edges.append([parent_idx, clu_idx])

        # Check if cluster is large enough or depth limit reached
        if len(idxs_clu) < min_imgs or depth >= max_depth:
            continue

        # Register cluster images to mean (optional - expensive!)
        if skip_registration and len(idxs_clu) >= 1000:
            imgs_reg = imgs_root[idxs_clu]
        else:
            tf_type = transform if len(idxs_clu) < 1000 else "translation"
            imgs_reg = register2mean(imgs_root[idxs_clu], tf_type)

        # Reduce dimensionality with PCA
        X = imgs_reg.reshape(len(imgs_reg), -1)
        svd_solver = "covariance_eigh" if X.shape[1] < 1000 else "full"
        n_pca = max(
            pca_level if isinstance(pca_level, int) else int(pca_level * X.shape[1]),
            num_tests,
        )
        Y = PCA(n_components=n_pca, svd_solver=svd_solver).fit_transform(X)

        # Normality tests on first num_tests components
        pvals = np.array([test(Y[:, k]) for k in range(min(num_tests, Y.shape[1]))])

        # Split cluster if not normal
        if np.any(pvals < pval_thr):
            n_gmm = (
                pca_level
                if isinstance(pca_level, int)
                else int(pca_level * X.shape[1])
            )
            Y_gmm = Y[:, :n_gmm]
            Z = torch.tensor(Y_gmm, dtype=torch.float32, device=device)
            _, _, _, idx, _, _ = train_GMM(Z, 2, seed=seed, verbose=False)
            clu_split = np.argmax(idx.cpu().numpy(), axis=0)

            # Ensure two clusters were found
            if np.any(clu_split == 0) and np.any(clu_split == 1):
                pile.append((clu_idx, idxs_clu[clu_split == 1], depth + 1))
                pile.append((clu_idx, idxs_clu[clu_split == 0], depth + 1))

    # Build adjacency and find leaves
    edges = np.array(edges) if edges else np.zeros((0, 2), dtype=int)
    adjacency = np.zeros((len(subclusters), len(subclusters)), dtype=bool)
    if len(edges) > 0:
        adjacency[edges[:, 0], edges[:, 1]] = True

    is_leaf = ~adjacency.any(axis=1)
    leaves = [
        subclusters[i]
        for i in range(len(subclusters))
        if is_leaf[i] and len(subclusters[i]) >= min_imgs
    ]

    return leaves


def tree_refine(
    imgs: np.ndarray,
    clu_pred: np.ndarray,
    transform: str = "homothety",
    min_imgs: int = 20,
    pca_level: float | int = 9,
    num_tests: int = 9,
    test: str = "ad",
    pval_thr: float = 2 * norm.sf(2).item(),
    seed: int = 0,
    device: str = "cpu",
    n_jobs: int = 1,
    max_depth: int = 10,
    skip_registration: bool = True,
) -> np.ndarray:
    """Refine clusters by growing trees of subclusters.

    Args:
        imgs: Character images, shape (n, h, w).
        clu_pred: Initial cluster assignment, int array of shape (n,).
        transform: Transform type for registration.
        min_imgs: Minimum images in a cluster to split.
        pca_level: PCA components (int) or variance ratio (float).
        num_tests: PCA components for normality testing.
        test: Normality test: 'ad', 'dp', 'ks', or 'sw'.
        pval_thr: p-value threshold for normality test.
        seed: Random seed for GMM.
        device: PyTorch device.
        n_jobs: Number of parallel jobs.
        max_depth: Maximum tree depth.
        skip_registration: If True, skip expensive registration.

    Returns:
        New cluster assignment with -1 for unassigned (garbage).
    """
    clu_idxs = np.unique(clu_pred)
    cluster_masks = {idx: np.where(clu_pred == idx)[0] for idx in clu_idxs}

    # Grow trees in parallel
    trees = Parallel(n_jobs=n_jobs)(
        delayed(grow_tree)(
            imgs_root=imgs[cluster_masks[clu_idx]],
            transform=transform,
            min_imgs=min_imgs,
            pca_level=pca_level,
            num_tests=num_tests,
            test=TESTS[test],
            pval_thr=pval_thr,
            seed=seed,
            device=device,
            max_depth=max_depth,
            skip_registration=skip_registration,
        )
        for clu_idx in tqdm(clu_idxs, desc="Growing trees")
    )

    # Assemble trees
    clu_pred_new = np.full_like(clu_pred, -1)
    new_clu_id = 0

    for root_idx, leaves in tqdm(
        zip(clu_idxs, trees), desc="Picking fruit", total=len(clu_idxs)
    ):
        idxs_root = cluster_masks[root_idx]
        for leaf in leaves:
            clu_pred_new[idxs_root[leaf]] = new_clu_id
            new_clu_id += 1

    return clu_pred_new
