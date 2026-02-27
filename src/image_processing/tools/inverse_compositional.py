# An implementation of the Inverse Compositional algorithm adapted from [BFS18].
#
# Notation follows [BFS18, Algorithm 3].
#
# Simplifications:
# - delta set to 0: because our images are already very small (maximum 30x30).
# - Not multiscale: same reason.
# - No prefiltering: based on experiences in [BFS18, §4.2.3], the Central
#     Differences schemes performs bettter with little noise, which is our case
#     since the reference I1 will by the estimated mean of the Gaussian.
#
# Other changes:
# - I1 and I2 need not have the same shape.
# - jmax set to 100 instead of 30.
#
# Reference:
# [BFS18] Briand, Facciolo, Sanchez. Improvements of the Inverse Compositional
#     Algorithm for Parametric Motion Estimation. In IPOL, 2018


import numpy as np
from skimage.transform import (AffineTransform, EuclideanTransform,
                               SimilarityTransform, ProjectiveTransform, warp)
from skimage import transform as tf



def gradient(I: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute the discrete gradient of an image.

    Args:
        I: Array of shape (h, w)|(h, w, c).

    Returns:
        nablaI: Array [Iy, Ix] of shape (2, h, w)|(2, h, w, c), where Iy (resp.
            Ix) is the partial derivative along height (resp. width) direction.
    """
    nablaI = np.zeros((2,) + I.shape)

    Ipad = np.pad(I, pad_width=1, mode='edge')

    nablaI[0] = (Ipad[1:-1, 2:] - Ipad[1:-1, :-2]) / 2
    nablaI[1] = (Ipad[2:, 1:-1] - Ipad[:-2, 1:-1]) / 2

    return nablaI


def p_to_psi(p: tuple[float, ...], transform: str) -> ProjectiveTransform:
    """Convert a planar transformation from tuple of parameters to skimage
    transform object."""

    if transform == 'translation':
        tx, ty = p
        return SimilarityTransform(translation=[tx, ty])

    elif transform == 'euclidean': # Clockwise!
        tx, ty, theta = p
        return SimilarityTransform(rotation=theta, translation=[tx, ty])

    elif transform == 'homothety': # Not in [BFS2018]
        tx, ty, a = p
        return SimilarityTransform(scale=1+a, translation=[tx, ty])

    elif transform == 'similarity':
        tx, ty, a, b = p
        matrix = [[1+a,  -b, tx],
                  [  b, 1+a, ty],
                  [  0,   0,  1]]
        return SimilarityTransform(matrix=matrix)

    elif transform == 'affinity':
        tx, ty, a11, a12, a21, a22 = p
        matrix = [[1+a11,   a12, tx],
                  [  a21, 1+a22, ty],
                  [    0,     0,  1]]
        return AffineTransform(matrix=matrix)

    # TODO: uncomment when discard_pixels() implements the homography case
    # elif transform == 'homography':
    #     h11, h12, h13, h21, h22, h23, h31, h32 = p
    #     matrix = [[1+h11,   h12, h13],
    #               [  h21, 1+h22, h23],
    #               [  h31,   h32,  1 ]]
    #     return ProjectiveTransform(matrix=matrix)

    else:
        raise ValueError()


def matrix_to_p(A: np.ndarray, transform: str) -> tuple[float, ...]:
    """Convert a planar transformation from matrix (in projective coordinates)
    to tuple of parameters."""

    if A.shape != (3, 3):
        raise ValueError(f'Input matrix A is of shape {A.shape};'+
                         f' shape (3, 3) expected.')

    if transform == 'translation':
        return A[0, 2], A[1, 2]

    elif transform == 'euclidean':
        return A[0, 2], A[1, 2], np.arctan2(A[1, 0], A[1, 1])

    elif transform == 'homothety':
        return A[0, 2], A[1, 2], A[0, 0] - 1

    elif transform == 'similarity':
        return A[0, 2], A[1, 2], A[0, 0] - 1, A[1, 0]

    elif transform == 'affinity':
        return A[0, 2], A[1, 2], A[0, 0] - 1, A[0, 1], A[1, 0], A[1, 1] - 1

    # TODO: uncomment when discard_pixels() implements the homography case
    # elif transform == 'homography':
    #     return (A[0, 0] - 1, A[0, 1],     A[0, 2],
    #             A[1, 0],     A[1, 1] - 1, A[1, 2],
    #             A[2, 0],     A[2, 1])

    else:
        raise ValueError()


def jacobian(shape: tuple[int], transform: str) -> np.ndarray:
    """Map the Jacobians of assotiated to a given Lie group of transformations.

    More precisely, compute, for each point P of the plane, the Jacobian around
    0 of the function which maps a tuple parametrizing a transformation Psi,
    to Psi(P).

    Args:
        shape: Shape of the domain where we want to map the Jacobians (i.e.
            I.shape for any image I on this domain).
        transform: Name of the type of transformations considered.

    Returns:
        A batch of matrices as an array of shape (h, w, 2, p) where (h, w) is
        the input shape and p is the number of parameters of the transform.
    """
    yy, xx = np.indices(shape)

    if transform == 'translation':
        J = np.zeros(shape + (2, 2))
        J[:, :, :, :] = np.eye(2)

    elif transform == 'euclidean': # Clockwise!
        J = np.zeros(shape + (2, 3))
        J[:, :, :, :2] = np.eye(2)
        J[:, :, 0, 2] = -yy
        J[:, :, 1, 2] = xx

    elif transform == 'homothety': # Not in [BFS18]
        J = np.zeros(shape + (2, 3))
        J[:, :, :, :2] = np.eye(2)
        J[:, :, 0, 2] = xx
        J[:, :, 1, 2] = yy

    elif transform == 'similarity':
        J = np.zeros(shape + (2, 4))
        J[:, :, :, :2] = np.eye(2)
        J[:, :, 0, 2] = xx
        J[:, :, 0, 3] = -yy
        J[:, :, 1, 2] = yy
        J[:, :, 1, 3] = xx

    elif transform == 'affinity':
        J = np.zeros(shape + (2, 6))
        J[:, :, :, :2] = np.eye(2)
        J[:, :, 0, 2] = xx
        J[:, :, 0, 3] = yy
        J[:, :, 1, 4] = xx
        J[:, :, 1, 5] = yy

    # TODO: uncomment when discard_pixels() implements the homography case
    # elif transform == 'homography':
    #     J = np.zeros(shape + (2, 8))
    #     J[:, :, 0, 0] = xx
    #     J[:, :, 0, 1] = yy
    #     J[:, :, 0, 2] = 1
    #     J[:, :, 0, 6] = -xx**2
    #     J[:, :, 0, 7] = -xx*yy
    #     J[:, :, 1, 3] = xx
    #     J[:, :, 1, 4] = yy
    #     J[:, :, 1, 5] = 1
    #     J[:, :, 1, 6] = -xx*yy
    #     J[:, :, 1, 7] = -yy**2

    else:
        raise ValueError()

    return J


def discard_pixels(I1shape: tuple[int, int],
                   I2shape: tuple[int, int],
                   Psi: EuclideanTransform | AffineTransform
                   ) -> np.ndarray:
    """ Discards pixels of Omega1 outside of Psi^{-1}(Omega2).

    It only works when Psi is an affine transformation!

    Args:
        I1shape: I1.shape, for any image I1 of domain Omega1
        I2shape: I2.shape, for any image I2 of domain Omega2
        Psi: skimage representation of Psi

    Returns:
        A bool array of shape I1shape, where False indicates discarded pixels.
    """

    # TODO: handle the homography case, i.e. allow any ProjectiveTransform
    if (not isinstance(Psi, EuclideanTransform)
        and not isinstance(Psi, AffineTransform)):
        raise ValueError()

    ny, nx = I2shape # In inverse_compositional() comments, (ny, nx) is I1.shape

    #       [[a, b, c],
    # Psi =  [d, e, f],
    #        [0, 0, 1]]
    a, b, c = Psi.params[0]
    d, e, f = Psi.params[1]

    # Compute Psi_x(x, y), Psi_y(x, y)
    yy, xx = np.indices(I1shape)
    psix = a*xx + b*yy + c
    psiy = d*xx + e*yy + f

    return (psix >= 0) * (psix <= nx-1) * (psiy >= 0) * (psiy <= ny-1)


def inverse_compositional(I1: np.ndarray,
                          I2: np.ndarray,
                          p0: tuple[float, ...],
                          transform: str = 'homothety',
                          epsilon: float = 0.001,
                          jmax: int = 100
                          ) -> tuple[float, ...]:
    """An implementation of Inverse Compositional adapted from [BFS18].

    Notation follows [BFS18, Algorithm 3].

    Args:
        I1: Array of shape (h_1, w_1)|(h_1, w_1, c), the reference image.
        I2: Array of shape (h_2, w_2)|(h_2, w_2, c), the image to align.
        p0: Parameters of the initial guess for the transformation.
        transform: Name of the type of transformations considered. Accepted
            values are the same as for `jacobian()`.
        epsilon: End iterations when the norm of increments falls below this.
        jmax: Maximum number of iterations.

    Returns:
        Parameters of the transformation Psi such that I1(.) = I2(Psi(.)), i.e.
        I1 == skimage.transforms.warp(I2, Psi, output_shape=I1.shape)
    """

    ### Grayscale conversion
    if len(I1.shape) == 3:
        I1 = I1.mean(axis=-1)
        I2 = I2.mean(axis=-1)

    ### Precomputations

    # Gradient
    nablaI1 = gradient(I1) # shape (2, ny, nx)

    # Jacobian
    J = jacobian(I1.shape, transform=transform) # shape (ny, nx, 2, np)

    # G, G.T @ G
    G = np.einsum('yxij,iyx->yxj', J, nablaI1) # shape (ny, nx, np)
    GTG = np.einsum('yxi,yxj->yxij', G, G) # shape (ny, nx, np, np)

    ### Incremental refinement

    # Initialize
    p = p0
    j = 1
    increment_size = epsilon + 1

    # Let's gooo
    while j <= jmax and increment_size > epsilon:
        Psi = p_to_psi(p, transform=transform)

        # Discard pixels outside boundary
        mask = discard_pixels(I1.shape, I2.shape, Psi) # shape (ny, nx)

        # DI
        DI = warp(I2, Psi, output_shape=I1.shape, mode='edge', order=3) - I1 # shape (ny, nx)

        # TODO: compute rhop

        # b, H
        b = np.einsum('yx,yxi,yx->i', mask, G, DI) # shape (np,)
        H = np.einsum('yx,yxij->ij', mask, GTG) # shape (np, np)

        # Delta_p
        deltap = np.linalg.solve(H, b)

        # Update p
        deltaPsi = p_to_psi(deltap, transform=transform)
        matrix = Psi.params @ deltaPsi.inverse.params
        p = matrix_to_p(matrix, transform=transform)

        # Manage while loop
        increment_size = np.linalg.norm(deltap)
        j += 1

    return p


def register2ref(img: np.ndarray,
                 ref: np.ndarray,
                 transform: str = 'homothety',
                 return_tf_matrix: bool = False
                 ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Register an image to a reference.

    If the registration fails (ill-posed problem: too many degrees of freedom
    causing linear system to be singular), it falls back to the translation-only
    case. If this fails too, the function returns the original image.

    Args:
        img: Input image, shape (h, w)|(h, w, c).
        ref: Reference image, shape (h, w)|(h, w, c).
        transform: Name of the type of transformations considered. Accepted
            values are the same as for `jacobian()`.
        return_tf_matrix: If True, returns the map from output to input domain.

    Returns:
        - Registered image, shape (h, w)|(h, w, c).
        - (optional, if return_tf_matrix is True)
          Projective matrix mapping output to input domain, shape (3, 3).
    """
    # Initialize p0
    p0 = matrix_to_p(np.eye(3), transform=transform)
    try:
        # Compute p using downsampled images
        p = inverse_compositional(ref, img, p0, transform=transform)
        # Apply transformation to the original image
        psi = p_to_psi(p, transform)
        if return_tf_matrix:
            return tf.warp(img, psi, order=3), psi.params
        else:
            return tf.warp(img, psi, order=3)
    # If the previous method fails, try with translation only
    except np.linalg.LinAlgError:
        if transform != 'translation':
            return register2ref(img, ref, 'translation', return_tf_matrix)
        else:
            return (img, np.eye(3)) if return_tf_matrix else img


def register2mean(imgs: np.ndarray,
                  transform: str = 'homothety',
                  return_tf_matrix: bool = False
                  ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Register a batch of images to their mean.

    Args:
        imgs: Batch of input images, shape (n, h, w)|(n, h, w, c).
        transform: Name of the type of transformations considered. Accepted
            values are the same as for `jacobian()`.
        return_tf_matrix: If True, returns the maps from output to input domain.

    Returns:
        - Registered images, shape (n, h, w)|(n, h, w, c).
        - (optional, if return_tf_matrix is True)
          Projective matrices mapping output to input domain, shape (n, 3, 3).
    """
    ref = imgs.mean(axis=0)
    tf_imgs = np.zeros(imgs.shape)
    if return_tf_matrix:
        psi_mats = np.zeros((imgs.shape[0], 3, 3))
    for ix, img in enumerate(imgs):
        if return_tf_matrix:
            tf_imgs[ix], psi_mats[ix] = register2ref(img, ref, transform,
                                                     return_tf_matrix)
        else:
            tf_imgs[ix] = register2ref(img, ref, transform, return_tf_matrix)
    return (tf_imgs, psi_mats) if return_tf_matrix else tf_imgs


def register_batch2ref(imgs: np.ndarray,
                       ref: np.ndarray,
                       transform: str = 'homothety',
                       return_tf_matrix: bool = False
                       ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Register a batch of images to a given reference.

    Args:
        imgs: Batch of input images, shape (n, h, w)|(n, h, w, c).
        ref: Reference image, shape (h, w)|(h, w, c).
        transform: Name of the type of transformations considered. Accepted
            values are the same as for `jacobian()`.
        return_tf_matrix: If True, returns the maps from output to input domain.

    Returns:
        - Registered images, shape (n, h, w)|(n, h, w, c).
        - (optional, if return_tf_matrix is True)
          Projective matrices mapping output to input domain, shape (n, 3, 3).
    """
    tf_imgs = np.zeros(imgs.shape)
    if return_tf_matrix:
        psi_mats = np.zeros((imgs.shape[0], 3, 3))
    for ix, img in enumerate(imgs):
        if return_tf_matrix:
            tf_imgs[ix], psi_mats[ix] = register2ref(img, ref, transform,
                                                     return_tf_matrix)
        else:
            tf_imgs[ix] = register2ref(img, ref, transform, return_tf_matrix)
    return (tf_imgs, psi_mats) if return_tf_matrix else tf_imgs