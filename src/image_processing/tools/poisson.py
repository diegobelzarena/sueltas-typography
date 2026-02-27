### Poisson editing, following Enric Meinhardt-Llopis @mnhrdt
#
# The image is discretised as the graph (V,E) of the rectangular grid: each
# {interior,edge,corner} pixel is a vertex of degree {4,3,2}.
#
# The gradient is then a matrix |E| x |V|
# The divergence is minus its transpose, a matrix |V| x |E|
# The laplacian is the divergence of the gradient, a matrix |V| x |V|


import numpy as np
from scipy.sparse import csr_array, diags_array, eye_array, kron, vstack
from scipy.sparse.linalg import spsolve


def discrete_gradient(h: int, w: int) -> csr_array:
    """Matrix of the gradient operator for a rectangular domain."""
    x = eye_array(w-1, w, k=1) - eye_array(w-1, w)  # path graph of length W
    y = eye_array(h-1, h, k=1) - eye_array(h-1, h)  # path graph of length H
    p = kron(eye_array(h), x, format='csr')         # H horizontal paths
    q = kron(y, eye_array(w), format='csr')         # W vertical paths
    B = vstack([p, q])                              # union of all paths
    return B


def solve_poisson(f: np.ndarray,
                  g: np.ndarray,
                  m: np.ndarray
                  ) -> np.ndarray:
    """Solve the Poisson equation: find an image u such that
            Δu = f  where m
                u = g  where not m
    (solution by local discrete method).

    Args:
        f: target laplacian, shape (h, w)
        g: boundary condition, shape (h, w)
        m: domain mask, shape (h, w)

    Returns:
        u: solution image, shape (h, w)
    """
    # flatten the images into vectors
    h, w = f.shape
    f = f.flatten()
    g = g.flatten()
    m = m.flatten() * 1.0

    # state and solve the linear system
    B = discrete_gradient(h, w)            # gradient operator
    L = -B.T @ B                           # laplacian operator
    M = diags_array(m, format='csr')       # mask operator
    I = eye_array(h*w, h*w, format='csr')  # identity operator
    A = (I - M)     - M @ L                # linear system: matrix
    b = (I - M) @ g - M @ f                # linear system: constant terms
    z = spsolve(A, b)                      # linear system: solution
    u = z.reshape(h, w)                    # recover a 2D array from the solution vector
    return u


def edit_poisson(f: np.ndarray,
                 g: np.ndarray,
                 m: np.ndarray
                    ) -> np.ndarray:
    """Poisson-edit an image (source) into another (destination) inside a mask.

    Args:
        f: source image, shape (h, w) | (h, w, c)
        g: destination image, shape (h, w) | (h, w, c)
        m: copying mask, shape (h, w)

    Returns:
        output image, shape (h, w) | (h, w, c)
    """
    # if image is color, call the gray-scale version recursively
    if len(f.shape) == 3:
        return np.stack([edit_poisson(f[..., c], g[..., c], m)
                            for c in range(f.shape[2])], axis=2)

    # flatten the images into vectors
    h, w = f.shape
    f = f.flatten()
    g = g.flatten()
    m = m.flatten() * 1.0

    # build linear operators
    B = discrete_gradient(h, w)  # gradient operator

    # compute gradients of each image
    nf = B @ f                   # gradient of source image
    ng = B @ g                   # gradient of destination image

    # compute the target laplacian x
    dm = (B @ m != 0)               # boundary of the mask (edge mask)
    x = -B.T @ ((1-dm)*nf + dm*ng)  # divergence of the combined gradient

    # recover the image from this gradient
    return solve_poisson(x.reshape(h, w), g, m)