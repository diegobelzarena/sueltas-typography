import torch
from tqdm import trange


## Initial cluster estimation with k-means++


def kmeans_init(X: torch.Tensor,
                k: int,
                seed: int | None = None):
    """K-means++ initialization of cluster centers for k-means.

    Args:
        X: Data points, shape (n, d).
        k: Number of clusters.
        seed: Random seed for reproducibility.
    
    Returns:
        idx: Index of the closest cluster center for each point, shape (n,).
    """
    n, d = X.shape
    if seed is not None:
        torch.manual_seed(seed)
    mu = torch.zeros((k, d), device=X.device)
    idx = torch.zeros((n,), dtype=torch.long, device=X.device)
    # Randomly select the first center among data points
    i = torch.randint(0, n, (1,)).item()
    mu[0] = X[i]
    # Subsequent centers are selected with a probability proportional to the
    # distance to the nearest center
    D = torch.sum((X - mu[0])**2, axis=1) # Squared distance to the nearest center
    for i in range(1, k):
        # Select the next center
        p = D/torch.sum(D)
        j = torch.multinomial(p, num_samples=1).item()
        mu[i] = X[j]
        # Update distances and assignments
        Di = torch.sum((X - mu[i])**2, axis=1)
        reass = Di < D
        idx[reass] = i
        D[reass] = Di[reass]
    return idx


def kmeans_ass(X: torch.Tensor,
               m: torch.Tensor
               ) -> tuple[torch.Tensor, torch.Tensor]:
    """Assignment step of the k-means algorithm.

    Args:
        X: Data points, shape (n, d).
        m: Cluster means, shape (k, d).

    Returns:
        idx: Index of the closest cluster mean for each point, shape (n,).
        D: Squared distance to the closest cluster mean, shape (n,).
    """
    k, _ = m.shape
    n, _ = X.shape
    # Initialize: assign everything to the first mean
    D = torch.sum((X - m[0])**2, axis=1) # distance to the first mean
    idx = torch.zeros((n,), dtype=torch.long, device=X.device)
    # Mean by mean, reassign points
    for i in range(1, k):
        Di = torch.sum((X - m[i])**2, axis=1) # distance to the i-th mean
        reass = Di < D
        idx[reass] = i
        D[reass] = Di[reass]
    return idx, D


def kmeans_upd(X: torch.Tensor, idx: torch.Tensor, k: int) -> torch.Tensor:
    """Update step of the k-means algorithm.
    
    Args:
        X: Data points, shape (n, d).
        idx: Cluster assignments, shape (n,) and dtype int.
        k: Number of clusters.
        
    Returns:
        Cluster means, shape (k', d) where k'<=k is the number of non-empty
        clusters.
    """
    _, d = X.shape
    # Cluster sizes π_i
    Pi = torch.zeros((k,), device=X.device, dtype=torch.long)
    vals, counts = torch.unique(idx, return_counts=True)
    Pi[vals] = counts
    # Cluster means
    mu = torch.zeros((k, d), device=X.device)
    mu = mu.index_add(0, idx, X)/Pi[:, None]
    # Discard empty clusters
    idx_disc = (Pi == 0)
    if idx_disc.any():
        mu = mu[~idx_disc]
    return mu


## Covariance matrix utilities


def regularization(Cov: torch.Tensor, ns: torch.Tensor) -> torch.Tensor:
    """Regularize covariance matrices with OAS (Oracle Approximating Shrinkage).
    
    Args:
        Cov: Batch of covariance matrices, shape (k, d, d).
        ns: Sizes of samples used to estimate the covariances, shape (k,).
        
    Returns:
        Regularized covariance matrices, shape (k, d, d).
    """
    _, d, _ = Cov.shape
    # Trace(Cov)
    TrS = Cov.diagonal(offset=0, dim1=-1, dim2=-2).sum(1) # shape (k,)
    # Trace(Cov^2)
    TrS2 = torch.sum(Cov**2, axis=(1, 2)) # shape (k,)
    # Shrinkage factor
    rho = ((1 - 2/d) * TrS2 + TrS**2) / ((ns + 1 - 2/d) * (TrS2 - TrS**2/d))
    rho[torch.isnan(rho)] = 0
    rho = rho.clip(0., 1.)[:, None, None] # Also handles inf
    # Regularized covariance matrix
    Cov = ( (1-rho) * Cov
           + rho * TrS[:, None, None]/d * torch.eye(d, device=Cov.device) )
    # Add an epsilon identity to degenerate matrices
    Cov[rho.squeeze()==0] += 1e-5 * torch.eye(d, device=Cov.device)
    return Cov


def cholesky(Cov: torch.Tensor, gamma: float = 1e-6) -> torch.Tensor:
    """Cholesky decomposition of covariance matrices with error handling."""
    _, d, _ = Cov.shape # shape (k, d, d)
    L, info = torch.linalg.cholesky_ex(Cov)
    # Handle errors: non SPD matrices
    idx_err = L.isnan().any(dim=(1, 2)) | (info > 0)
    a = 1
    while idx_err.any():
        # Add a small multiple of the identity matrix and try again
        Cov[idx_err] += a * gamma * torch.eye(d, device=Cov.device)
        L[idx_err], info_idx = torch.linalg.cholesky_ex(Cov[idx_err])
        # Handle remaining errors
        info = torch.zeros_like(info)
        info[idx_err] = info_idx
        idx_err = L.isnan().any(dim=(1, 2)) | (info > 0)
        a*=10
    return L


## GMM fitting with EM


# def M_step(X:torch.Tensor,
#            idx: torch.Tensor,
#            k: int
#            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#     """Maximisation step of EM for GMM fitting.
    
#     Args:
#         X: Data points, shape (n, d).
#         idx: Cluster assignments, shape (n,) and dtype int.
#         k: Number of clusters.
        
#     Returns:
#         GMM parameters: with k' <= k,
#         - Pi: Gaussian mixture probabilities, shape (k',).
#         - mu: Gaussian means, shape (k', d).
#         - Cov: Gaussian covariances, shape (k', d, d).
#     """
#     n, d = X.shape
#     # Cluster sizes π_i
#     Pi = torch.zeros((k,), device=X.device, dtype=torch.long)
#     vals, counts = torch.unique(idx, return_counts=True)
#     Pi[vals] = counts
#     # Cluster means
#     mu = torch.zeros((k, d), device=X.device)
#     mu = mu.index_add(0, idx, X)/Pi[:, None]
#     # Cluster covariances
#     Cov = torch.zeros((k, d, d), device=X.device)
#     Y = X - mu[idx]
#     Cov = Cov.index_add(0, idx, torch.einsum('ni,nj->nij', Y, Y)
#                         )/Pi[:, None, None]
#     # # Alternative to previous line;
#     # # einsum+index_add speed sensitive to n or dtype, not to k
#     # # matmul+forloop speed sensitive to k, not to n nor dtype
#     # # Both are equivalent around k=1e3, n=1e4, float32
#     # #                  or around k=1e3, n=5e3, float64
#     # for i in vals:
#     #     Zi = Z[idx == i]
#     #     Cov[i] = Zi.T @ Zi / Pi[i]

#     # Discard (almost) empty clusters
#     idx_disc = (Pi < 2)
#     if idx_disc.any():
#         Pi = Pi[~idx_disc]
#         mu = mu[~idx_disc]
#         Cov = Cov[~idx_disc]

#     Pi = Pi.type(torch.float32)/n
#     return Pi, mu, Cov


def M_step(X: torch.Tensor,
           w: torch.Tensor
           ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Maximisation step of EM for GMM fitting, with OAS-regularized covariance.
    
    Args:
        X: Data points, shape (n, d).
        w: Classification weights w[i, j] = P(Gaussian i | X_j), shape (k, n).
        
    Returns:
        GMM parameters: with k' <= k,
        - Pi: Gaussian mixture probabilities, shape (k',).
        - mu: Gaussian means, shape (k', d).
        - Cov: Gaussian covariances, shape (k', d, d).
    """
    _, d = X.shape
    # Mixture probabilities π_i; unnormalized for now
    Pi = w.sum(axis=1) # shape (k,)
    # Discard (almost) empty clusters
    idx_keep = (Pi >= 2)
    Pi = Pi[idx_keep]
    w = w[idx_keep]
    k1, _ = w.shape
    # Cluster means
    mu = (w/Pi[:, None]) @ X # shape (k, d)
    # Cluster covariances; MLE because it will be regularized with OAS
    Cov = torch.zeros((k1, d, d), device=X.device)
    for i in range(0, k1):
        idx_i = w[i] > 0 # shape (n,), sum ni
        wi = w[i, idx_i] # shape (ni,)
        Y = X[idx_i] - mu[i] # shape (ni, d)
        Cov[i] = (wi[:, None]*Y).T @ Y / Pi[i] # shape (d, d)
    # Normalize mixture probabilities
    Pi /= Pi.sum()
    # Regularize covariance matrices with OAS
    Cov = regularization(Cov, Pi*len(X))
    return Pi, mu, Cov


def E_step(X: torch.Tensor,
           Pi: torch.Tensor,
           mu: torch.Tensor,
           Cov: torch.Tensor
           ) -> tuple[torch.Tensor, torch.Tensor]:
    """Expectation step of EM for GMM fitting.
    
    Args:
        X: Data points, shape (n, d).
        Pi: Gaussian mixture probabilities, shape (k,).
        mu: Gaussian means, shape (k, d).
        Cov: Gaussian covariances, shape (k, d, d).
    Returns:
        w: Classification weights w[i, j] = P(Gaussian i | X_j), shape (k, n).
        logLH: Pointwise log-likelihood of the GMM, shape (n,).
    """
    n, d = X.shape
    k = len(Pi)
    # Cholesky decomposition: Cov = L@L.T, with L lower triangular
    L = cholesky(Cov) # shape (k, d, d)
    Linv = torch.linalg.solve_triangular(L, torch.eye(d, device=X.device),
                                         upper=False) # shape (k, d, d)
    # Log determinant of Cov
    inds = torch.linspace(0, d-1, d, dtype=torch.long)
    logDetS = -2 * torch.sum(torch.log(Linv[:, inds, inds]), axis=1) # shape (k,)
    # logP[i, j] = log( P(X_j | Gaussian i) * P(Gaussian i) )
    logP = torch.zeros((k, n), device=X.device)
    for i in range(0, k):
        logP[i] = ( - 0.5 * ((Linv[i] @ (X-mu[i]).T)**2).sum(axis=0)
                    - 0.5 * logDetS[i]
                    - 0.5 * d * torch.log(torch.tensor(2*torch.pi))
                    + torch.log(Pi[i]) )
    # logLH[j] = log P(X_j)
    logLH = torch.logsumexp(logP, axis=0) # shape (n,)
    # w[i, j] = P(Gaussian i | X_j)
    w = torch.exp(logP-logLH)

    return w, logLH


def train_GMM(X: torch.Tensor,
              n_clusters: int,
              max_steps: int = 100,
              epsilon: float = 1e-6,
              seed: int = None,
              verbose: bool =False
              ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                         list[float], list[float]]:
    """Train a Gaussian Mixture Model (GMM) using EM initialized with k-means++.
    
    Args:
        X: Data points, shape (n, d).
        n_clusters: Number of clusters (GMM components).
        max_steps: Maximum number of iterations.
        epsilon: Convergence threshold.
        seed: RNG seed for reproducibility.
        verbose: If True, show progress bars.
        
    Returns:
        Pi: Gaussian mixture probabilities, shape (k,), where k <= n_clusters is
            the final number of clusters.
        mu: Gaussian means, shape (k, d).
        Cov: Gaussian covariances, shape (k, d, d).
        w: Classification weights w[i, j] = P(Gaussian i | X_j), shape (k, n).
        logLHs: List of log-likelihoods at each EM iteration.
        Ds: List of distances at each k-means iteration.
    """
    # If all points equal, return only one cluster
    if (X == X[0]).all():
        return ( torch.tensor([1.]), X[0], torch.eye(X.shape[1]),
                 torch.ones((1, len(X)), dtype=torch.long), [0.], [0.] )
    
    # Initialize the cluster centers with the k-means++ algorithm
    idx = kmeans_init(X, n_clusters, seed)
    Ds = []
    loop = (trange(max_steps, desc='GMM estimation: KNN initialisation')
            if verbose else range(max_steps))
    k = n_clusters
    for t in loop:
        mu = kmeans_upd(X, idx, k)
        k = len(mu)
        idx, D = kmeans_ass(X, mu)
        Ds.append(D.sum().item())
        # Stopping criterion: relative decrease in loss
        if t >= 1:
            l0, l1 = torch.tensor(Ds[-2:])
            if ((l0-l1)/l0).abs() < epsilon:
                break

    # Run the EM algorithm with regularized covariance matrices
    w = torch.eye(n_clusters, device=X.device)[:, idx]
    logLHs = []
    loop = (trange(max_steps, desc='GMM estimation: EM')
            if verbose else range(max_steps))
    for t in loop:
        Pi, mu, Cov = M_step(X, w)
        w, logLH = E_step(X, Pi, mu, Cov)
        logLHs.append(logLH.sum().item())
        # Stopping criterion: relative decrease in loss
        if t >= 1:
            l0, l1 = torch.tensor(logLHs[-2:])
            if ((l0-l1)/l0).abs() < epsilon:
                break
    Pi, mu, Cov = M_step(X, w)

    return Pi, mu, Cov, w, logLHs, Ds
