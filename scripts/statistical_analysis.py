import numpy as np
from scipy.cluster.hierarchy import linkage, optimal_leaf_ordering, leaves_list
from scipy.spatial.distance import squareform
from scipy.stats import binom



def estimate_all_quantiles_(ds: np.ndarray,
                       alphas: np.ndarray,
                       inds_train: tuple[np.ndarray, np.ndarray] | None = None
                       ) -> np.ndarray:
    """Estimate quantile of the distribution of distances d_s(b_1, b_2) for each
    symbol s, conditionally on b_1 and b_2 not using the same type for s.

    Args:
        ds: Float array of distances for each symbol, of shape (ns, nb, nb)
            where ns is the number of symbols and nb is the number of books.
            Value NaN should be used when the distance is undefined.
        alphas: Probability thresholds.
        inds_train: 2-uple of int arrays with values in [0, nb) representing
            pairs of (indices of) books not know to be from the same printer.
            If None, all pairs of books are used.
    
    Returns:
        qs: Float array of shape (len(alphas), ns) with the estimated quantiles
            for each symbol.
    """
    ns, nb, _ = ds.shape
    if inds_train is None:
        inds_train = np.triu_indices(nb, k=1)
    qs = np.zeros((len(alphas), ns))
    for idx, d in enumerate(ds):
        vals = d[inds_train]
        vals = vals[~np.isnan(vals)]
        qs[:, idx] = np.quantile(vals, alphas)
    return qs


def acontrario_n1_all(ds: np.ndarray,
                  qs: np.ndarray,
                  alphas: np.ndarray,
                  N: int | None = None,
                  epsilon : float = 0.01
                  )-> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate n1 as in algorithm 1 of the paper.

    Args:
        ds: Float array of distances for each symbol, of shape (ns, nb, nb)
            where ns is the number of symbols and nb is the number of books.
            Value NaN should be used when the distance is undefined.
        qs: Float array of shape (len(alphas), ns,) with the estimated quantiles
            of level alpha for each symbol.
        alphas: Probability thresholds.
        N: Number of tests in NFA. If None, computed as number of pairs of books
            given in ds.
        epsilon: NFA threshold for the a contrario test.
    
    Returns:
        n: Int array of shape (nb, nb), number of symbols for which distance is
            defined.
        n1: Int array of shape (len(alphas), nb, nb).
        c: Bool array of shape (len(alphas), ns, nb, nb), indicates candidate
            symbols: those for which the distance is lower than threshold
            (quantile).
    """
    _, nb, _ = ds.shape
    if N is None:
        N = nb*(nb-1)//2

    n = np.isfinite(ds).sum(axis=0) # shape (nb, nb)
    c = ds < qs[..., None, None] # candidates: shape (len(alphas), ns, nb, nb)
    k = np.sum(c, axis=1) # shape (len(alphas), nb, nb)

    aa, kk, nn = np.meshgrid(alphas,
                             np.arange(-1, k.max()),
                             np.arange(n.max()+1),
                             indexing='ij')
    # survival function has strict inequality: P(Z > k), hence why kk is shifted
    probs = binom.sf(kk, nn, aa) # shape (len(alphas), kmax+1, nmax+1)

    n1 = np.arange(n.max() + 1) # shape (nmax+1,)
    NFA = N*probs[np.arange(len(alphas))[:, None, None, None],
                  np.maximum(k[..., None]-n1, 0),
                  np.maximum(n[..., None]-n1, 0)] # shape (len(alphas), nb, nb, nmax+1)
    return n, np.argmax(NFA > epsilon, axis=-1), c # shape (len(alphas), nb, nb)


def final_outputs(n1, c):
    inds = np.indices(n1.shape[1:])
    bestalpha = n1.argmax(axis=0)
    # Output: maximum n1 across alphas
    n1max = n1[bestalpha, *inds] # shape (nb, nb)
    # Corresponding list of candidate symbols
    cands = c[bestalpha, :, *inds].transpose((2, 0, 1)) # shape (ns, nb, nb)
    return n1max, cands


def hierarchical_olo_order(D, method='average'):
    """Hierarchical clustering + optimal leaf ordering. D = distance matrix."""
    # squareform expects zeros on diagonal; keep them
    cd = squareform(D, checks=False)
    Z = linkage(cd, method=method)
    Z_olo = optimal_leaf_ordering(Z, cd)
    return leaves_list(Z_olo).astype(int)