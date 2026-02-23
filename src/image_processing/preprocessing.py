import math
import numpy as np
from skimage.filters import threshold_sauvola
from skimage.morphology import diamond, dilation
import skimage.transform as skt
from tqdm import tqdm

from .tools.poisson import edit_poisson
from .tools.inverse_compositional import matrix_to_p


def bg_flatten(img_doc: np.ndarray,
               d: int = 0,
               equalize: bool = True,
               tile_size: int | None = None) -> np.ndarray:
    """Flatten the background of a document image; convert from 8-bit to float.
    
    Args:
        img_doc: 8-bit grayscale document image, shape (h, w).
        d: Dilation size for text mask.
        equalize: Whether to have consistent output contrast.
        tile_size: If not None, image is processed in tiles of this size.
        
    Returns:
        Image with background flattened to 1, shape (h, w) and values in [0, 1].
    """
    img = img_doc.astype(np.float32)/255.
    # Text mask
    thr = threshold_sauvola(img)
    mask_txt = dilation((img <= thr), diamond(d, decomposition='sequence'))
    # Tiling
    h, w = img.shape
    t = tile_size
    out = np.zeros((h, w), dtype=np.float32)
    if t is None:
        t = max(h, w)
    for i in range(0, h, t):
        for j in range(0, w, t):
            img_tile = img[i:i+t, j:j+t]
            out_tile = out[i:i+t, j:j+t]
            mask_tile = mask_txt[i:i+t, j:j+t]
            # Poisson editing
            out_tile[:, :] = edit_poisson(f=img_tile,
                                          g=np.ones_like(img_tile),
                                          m=mask_tile)
    # Equalization
    if equalize:
        dec = 1 - np.quantile(out[out<=1-1/255], 0.1)
        out = (out-1)*0.9/dec + 1
    # Clipping
    return np.clip(out, 0, 1)


def embed_noresize(imgs: list[np.ndarray],
          h: int = 32,
          w: int = 32,
          ) -> tuple[np.ndarray, np.ndarray]:
    
    max_h, max_w = h, w
    img_ready = np.zeros((len(imgs), max_h, max_w))
    tf_params = np.zeros((len(imgs), 3))
    for i, img in tqdm(enumerate(imgs),
                       desc="Processing characters", total=len(imgs)):
        h,w = img.shape
        if (h < max_h) and (w < max_w):
            n_img = np.ones((max_h,max_w))
            n_img[:h,:w] = img
            h,w = n_img.shape
            img = n_img.copy()
            
        cut_h, cut_w = [0,h], [0,w]
        img = 1-img
        normalized = img/(img.sum()+1e-7)
        
        ## Barycenter
        coords = np.indices(img.shape) # (2, h, w)
        bary = np.sum(normalized*coords, axis=(1, 2)) # (2,)

        center = np.array([h,w])/2
        transf = skt.SimilarityTransform(translation=(center-bary)[::-1])
        transformed_image = skt.warp(img, transf.inverse, order=3)
        # transformed_image-= transformed_image.min()
        # transformed_image/= (transformed_image.max() + 1e-7)
        
        h_lims = max(0, max_h//2 - h//2), min(max_h, max_h//2 + math.ceil(h/2))
        w_lims = max(0, max_w//2 - w//2), min(max_w, max_w//2 + math.ceil(w/2))
        if max_h < h:
            cut_h = max(0, h//2 - max_h//2), min(h, h//2 + math.ceil(max_h/2))
        if max_w < w:
            cut_w = max(0, w//2 - max_w//2), min(w, w//2 + math.ceil(max_w/2))
        img_ready[i, h_lims[0]:h_lims[1], w_lims[0]:w_lims[1]] = transformed_image[cut_h[0]:cut_h[1], cut_w[0]:cut_w[1]]
        tf_params[i] = matrix_to_p(transf.params, transform='homothety')

    return img_ready, tf_params
