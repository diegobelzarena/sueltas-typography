import numpy as np
from skimage.filters import threshold_sauvola
from skimage.morphology import diamond, dilation
from tools.poisson import edit_poisson


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




