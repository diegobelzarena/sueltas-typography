"""A contrario analysis for typographic distance matrices.

Core algorithms:
- ``load_adjacencies``: turn .npz adjacency data into usable distance tensors.
- ``estimate_all_quantiles_``: background‑model quantile estimation.
- ``acontrario``: NFA‑based detection (equation 9 of the paper).
- ``hierarchical_olo_order``: hierarchical clustering + optimal leaf ordering.
"""

from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import leaves_list, linkage, optimal_leaf_ordering
from scipy.spatial.distance import squareform
from scipy.stats import binom


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_adjacencies(data) -> np.ndarray:
    """Load adjacency matrices from an NPZ object and normalise sentinels.

    For each character the maximum value in the matrix marks pairs of documents
    that do *not* share the character.  This value is replaced by ``np.inf`` and
    the diagonal is set to zero for documents that contain the character.

    Parameters
    ----------
    data : numpy NpzFile
        Must contain an ``"adjacencies"`` key.

    Returns
    -------
    ds : np.ndarray, shape (n_symbols, n_docs, n_docs)
    """
    ds = data["adjacencies"].copy()
    for d in ds:
        x = d.max()
        inds, = np.nonzero(np.any(d != x, axis=0))
        d[d == x] = np.inf
        d[inds, inds] = 0
    return ds


# ---------------------------------------------------------------------------
# Quantile estimation
# ---------------------------------------------------------------------------

def estimate_all_quantiles_(
    ds: np.ndarray,
    alphas: np.ndarray,
    inds_train: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    """Estimate quantiles of the background distance distribution.

    For each symbol *s*, estimate the α‑quantile of *d_s(b₁, b₂)* conditionally
    on *b₁* and *b₂* **not** using the same type for *s*.

    Parameters
    ----------
    ds : np.ndarray, shape (n_symbols, n_docs, n_docs)
        Distance tensor.  Use ``np.inf`` for undefined distances.
    alphas : np.ndarray, shape (n_alpha,)
        Probability thresholds.
    inds_train : tuple[np.ndarray, np.ndarray] | None
        Pairs of documents to use for estimation.  If *None*, all upper‑triangle
        pairs are used.

    Returns
    -------
    qs : np.ndarray, shape (n_alpha, n_symbols)
    """
    ns, nb, _ = ds.shape
    if inds_train is None:
        inds_train = np.triu_indices(nb, k=1)
    qs = np.zeros((len(alphas), ns))
    for idx, d in enumerate(ds):
        vals = d[inds_train]
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            qs[:, idx] = np.inf
        else:
            qs[:, idx] = np.quantile(vals, alphas)
    return qs


def estimate_quantiles_cross_corpus(
    ds_target: np.ndarray,
    letters_target: np.ndarray,
    ds_ref: np.ndarray,
    letters_ref: np.ndarray,
    alphas: np.ndarray,
) -> np.ndarray:
    """Estimate background quantiles using a reference corpus where possible.

    For each letter in *letters_target*:

    * If the letter also appears in *letters_ref*, the quantile is estimated
      from the reference corpus distances (cross-corpus calibration).
    * Otherwise, the quantile is estimated from the target corpus itself.

    Parameters
    ----------
    ds_target : np.ndarray, shape (n_symbols_target, n_docs_target, n_docs_target)
        Distance tensor of the target corpus.
    letters_target : np.ndarray of str, shape (n_symbols_target,)
        Letter labels for each slice of *ds_target*.
    ds_ref : np.ndarray, shape (n_symbols_ref, n_docs_ref, n_docs_ref)
        Distance tensor of the reference corpus.
    letters_ref : np.ndarray of str, shape (n_symbols_ref,)
        Letter labels for each slice of *ds_ref*.
    alphas : np.ndarray, shape (n_alpha,)
        Probability thresholds.

    Returns
    -------
    qs : np.ndarray, shape (n_alpha, n_symbols_target)
    """
    # Build a lookup: letter -> index in the reference corpus
    ref_lookup = {letter: idx for idx, letter in enumerate(letters_ref)}

    ns_target = len(letters_target)
    qs = np.zeros((len(alphas), ns_target))

    n_from_ref = 0
    n_from_self = 0

    for t_idx, letter in enumerate(letters_target):
        r_idx = ref_lookup.get(letter)
        if r_idx is not None:
            # Use reference corpus distances for this letter
            d = ds_ref[r_idx]
            inds = np.triu_indices(d.shape[0], k=1)
            n_from_ref += 1
        else:
            # Letter not in reference — use target corpus itself
            d = ds_target[t_idx]
            inds = np.triu_indices(d.shape[0], k=1)
            n_from_self += 1

        vals = d[inds]
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            qs[:, t_idx] = np.inf
        else:
            qs[:, t_idx] = np.quantile(vals, alphas)

    return qs, n_from_ref, n_from_self


# ---------------------------------------------------------------------------
# A contrario detection
# ---------------------------------------------------------------------------

def acontrario(
    ds: np.ndarray,
    qs: np.ndarray,
    alphas: np.ndarray,
    N: int | None = None,
    epsilon: float = 0.01,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate *n̂₁* as in equation (9) of the paper.

    Parameters
    ----------
    ds : np.ndarray, shape (n_symbols, n_docs, n_docs)
        Distance tensor.
    qs : np.ndarray, shape (n_alpha, n_symbols)
        Background quantiles.
    alphas : np.ndarray, shape (n_alpha,)
        Probability thresholds.
    N : int | None
        Number of tests in the NFA.  Defaults to C(n_docs, 2).
    epsilon : float
        NFA detection threshold.

    Returns
    -------
    n : np.ndarray, shape (n_docs, n_docs)
        Number of symbols with a finite distance.
    n1hat : np.ndarray, shape (n_docs, n_docs)
        Estimated *n̂₁* values.
    cands : np.ndarray, shape (n_symbols, n_docs, n_docs)
        Boolean mask of candidate symbol matches at the best threshold.
    """
    _, nb, _ = ds.shape
    if N is None:
        N = nb * (nb - 1) // 2

    n = np.isfinite(ds).sum(axis=0)  # (nb, nb)
    c = ds < qs[..., None, None]      # (n_alpha, ns, nb, nb)
    k = np.sum(c, axis=1)             # (n_alpha, nb, nb)

    aa, kk, nn = np.meshgrid(
        alphas,
        np.arange(-1, k.max()),
        np.arange(n.max() + 1),
        indexing="ij",
    )
    probs = binom.sf(kk, nn, aa)  # (n_alpha, kmax+1, nmax+1)

    n1 = np.arange(n.max() + 1)
    P = probs[
        np.arange(len(alphas))[:, None, None, None],
        np.maximum(k[..., None] - n1, 0),
        np.maximum(n[..., None] - n1, 0),
    ]  # (n_alpha, nb, nb, nmax+1)

    # Best threshold: alpha minimizing P
    abest = P.argmin(axis=0)  # (nb, nb, nmax+1)
    p = P[abest, *np.indices(abest.shape)]  # (nb, nb, nmax+1)

    # Equation (9)
    n1hat = np.argmax(p >= epsilon / N, axis=-1)  # (nb, nb)

    # Candidate symbol matches at the best threshold
    cands = c[
        abest[*np.indices((nb, nb)), n1hat],
        :,
        *np.indices((nb, nb)),
    ].transpose((2, 0, 1))  # (ns, nb, nb)

    return n, n1hat, cands


# ---------------------------------------------------------------------------
# Ordering
# ---------------------------------------------------------------------------

def hierarchical_olo_order(
    D: np.ndarray,
    method: str = "average",
) -> np.ndarray:
    """Optimal‑leaf‑order permutation from a distance matrix.

    Parameters
    ----------
    D : np.ndarray, shape (n, n)
        Symmetric distance / dissimilarity matrix (zeros on diagonal).
    method : str
        Linkage method (``'average'``, ``'ward'``, …).

    Returns
    -------
    order : np.ndarray of int, shape (n,)
    """
    cd = squareform(D, checks=False)
    Z = linkage(cd, method=method)
    Z_olo = optimal_leaf_ordering(Z, cd)
    return leaves_list(Z_olo).astype(int)


# ---------------------------------------------------------------------------
# Alpha grid construction helper
# ---------------------------------------------------------------------------

def build_alpha_grid(
    base_factors: list[float] | None = None,
    exponent_range: list[int] | None = None,
) -> np.ndarray:
    """Build the sorted alpha grid used by the a contrario method.

    Parameters
    ----------
    base_factors : list[float]
        Base multipliers (default ``[0.1, 0.15]``).
    exponent_range : list[int]
        ``[low, high]`` inclusive range of exponents of 2 (default ``[-5, 3]``).

    Returns
    -------
    alphas : np.ndarray
    """
    if base_factors is None:
        base_factors = [0.1, 0.15]
    if exponent_range is None:
        exponent_range = [-5, 3]
    low, high = exponent_range
    alphas = sorted(
        b * 2**i for b in base_factors for i in range(low, high + 1)
    )
    alphas = np.array(alphas)
    alphas = alphas[(alphas > 0) & (alphas < 1)]
    return alphas
