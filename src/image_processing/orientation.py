"""
Local Orientation estimation method based on FFT radial profile analysis and structure tensor calculation.
"""
import numpy as np
import cv2
from skimage.filters import gaussian as gaussian_filter
from scipy import ndimage


def find_orientation_radial_profile(fft_magnitude):
    """
    Ultra-fast version using histogram binning
    """
    h, w = fft_magnitude.shape
    center_y, center_x = h // 2, w // 2

    # Pre-compute coordinate grids
    y, x = np.ogrid[:h, :w]
    x = x - center_x
    y = y - center_y

    # Calculate angles and convert to integer bins
    angles_deg = np.degrees(np.arctan2(y, x)) % 180
    angle_bins_int = np.round(angles_deg).astype(np.int32)

    # Filter valid angles (50-130 degrees)
    valid_mask = (angle_bins_int >= 70) & (angle_bins_int <= 110)

    if not np.any(valid_mask):
        return 90, np.array([]), np.array([])

    valid_angles = angle_bins_int[valid_mask]
    valid_magnitudes = fft_magnitude[valid_mask]

    # Use numpy's bincount for ultra-fast histogram computation
    angle_range = np.arange(70, 111)
    weighted_counts = np.bincount(
        valid_angles - 70,  # Shift to 0-based indexing
        weights=valid_magnitudes,
        minlength=41  # 131-50
    )
    pixel_counts = np.bincount(
        valid_angles - 70,
        minlength=41
    )

    # Compute mean magnitudes (avoid division by zero)
    radial_profile = np.divide(
        weighted_counts,
        pixel_counts,
        out=np.zeros_like(weighted_counts, dtype=np.float32),
        where=pixel_counts != 0
    )

    # Find dominant orientation
    if radial_profile.size > 0:
        dominant_angle_idx = np.argmax(radial_profile)
        dominant_orientation = angle_range[dominant_angle_idx]
    else:
        dominant_orientation = 90

    return dominant_orientation, radial_profile, angle_range

def local_fft_sliding_window(image, window_height=128, window_width=256, step_size=64):
    """
    Optimized sliding window FFT analysis with separate height and width

    Parameters:
    -----------
    image : numpy.ndarray
        Input grayscale image
    window_height : int
        Height of the sliding window
    window_width : int
        Width of the sliding window
    step_size : int or tuple
        Step size for sliding window. Can be:
        - int: same step for both dimensions
        - tuple (step_y, step_x): different steps for height and width
    """
    h, w = image.shape
    image = image / 255.0
    hp_img = cv2.GaussianBlur(image, (5, 5), sigmaX=11, sigmaY=11)
    hp_img = cv2.subtract(image, hp_img)
    image = hp_img.copy()

    # Handle different step sizes for x and y
    if isinstance(step_size, int):
        step_y = step_x = step_size
    else:
        step_y, step_x = step_size

    # Pad image to fit windows exactly
    pad_h = (window_height - h % window_height) % window_height
    pad_w = (window_width - w % window_width) % window_width
    image = np.pad(image, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
    h_padded, w_padded = image.shape

    # Pre-allocate arrays for better memory management
    max_windows = ((h_padded - window_height) // step_y + 1) * ((w_padded - window_width) // step_x + 1)
    orientations = np.zeros(max_windows, dtype=np.float32)
    positions = np.zeros((max_windows, 2), dtype=np.int32)

    window_count = 0

    # Use efficient iteration
    for y in range(0, h_padded - window_height + 1, step_y):
        for x in range(0, w_padded - window_width + 1, step_x):
            # Extract window
            window = image[y:y+window_height, x:x+window_width]

            # Skip if window is mostly empty (faster check)
            if window.std() < 0.00001:
                continue

            # Calculate FFT (use single precision for speed)
            f = np.fft.fft2(window.astype(np.float32))
            fshift = np.fft.fftshift(f)
            magnitude = np.abs(fshift)

            # Find orientation using ultra-fast method
            orientation, _, _ = find_orientation_radial_profile(magnitude)
            if (orientation % 90) <= 45:
                orientation = (orientation % 90)
            else:
                orientation = (orientation % 90 - 90)

            orientations[window_count] = orientation
            positions[window_count] = [x + window_width // 2, y + window_height // 2]
            window_count += 1


    # Trim arrays to actual size
    orientations = orientations[:window_count]
    positions = positions[:window_count]

    return orientations, positions


def structure_tensor(image, sigma=0.0, rho=0.3):
    """
    Calculate the structure tensor of an image

    Parameters:
    -----------
    image : numpy.ndarray
        Input grayscale image
    sigma : float
        Standard deviation for derivative calculation (noise scale)
    rho : float
        Standard deviation for tensor smoothing (integration scale)

    Returns:
    --------
    tensor_components : dict
        Dictionary containing:
        - 'Jxx': xx component of structure tensor
        - 'Jxy': xy component of structure tensor
        - 'Jyy': yy component of structure tensor
        - 'eigenvalues': (lambda1, lambda2) where lambda1 >= lambda2
        - 'coherence': measure of local orientation strength
        - 'orientation': local orientation angle in radians
    """

    # Ensure image is float
    if image.dtype != np.float64:
        image = image.astype(np.float64)

    # Calculate gradients using Gaussian derivatives
    # Sobel operators for gradient calculation
    grad_x = ndimage.sobel(image, axis=1)  # Gradient in x direction
    grad_y = ndimage.sobel(image, axis=0)  # Gradient in y direction

    # Alternative: Use Gaussian derivatives for better noise handling
    if sigma > 0:
        grad_x = gaussian_filter(image, sigma, order=[0, 1])
        grad_y = gaussian_filter(image, sigma, order=[1, 0])

    # Calculate tensor components
    Jxx = grad_x * grad_x
    Jxy = grad_x * grad_y
    Jyy = grad_y * grad_y

    # Smooth the tensor components (integration)
    if rho > 0:
        Jxx = gaussian_filter(Jxx, rho)
        Jxy = gaussian_filter(Jxy, rho)
        Jyy = gaussian_filter(Jyy, rho)

    # Calculate eigenvalues and derived measures
    trace = Jxx + Jyy
    det = Jxx * Jyy - Jxy * Jxy

    # Eigenvalues (lambda1 >= lambda2 >= 0)
    discriminant = np.sqrt((trace**2 - 4*det).clip(0))
    lambda1 = 0.5 * (trace + discriminant)
    lambda2 = 0.5 * (trace - discriminant)

    # Coherence measure (0 = isotropic, 1 = highly oriented)
    coherence = np.divide(lambda1 - lambda2, lambda1 + lambda2 + 1e-10)

    # Orientation angle (in radians)
    orientation = 0.5 * np.arctan2(2 * Jxy, Jxx - Jyy)

    return {
        'Jxx': Jxx,
        'Jxy': Jxy,
        'Jyy': Jyy,
        'eigenvalues': (lambda1, lambda2),
        'coherence': coherence,
        'orientation': orientation,
        'trace': trace,
        'determinant': det
    }

def filter_image_by_words(img, word_tblrs, padding=10):
    h, w = img.shape[:2]

    if not word_tblrs:
        return img

    # Create output array directly (zeros for non-word regions)
    filtered_img = np.zeros_like(img)

    # Process all words in one pass using direct array indexing
    for word in word_tblrs:
        t = max(0, word[0] - padding)
        b = min(h, word[1] + padding)
        l = max(0, word[2] - padding)
        r = min(w, word[3] + padding)

        filtered_img[t:b, l:r] = img[t:b, l:r]

    return filtered_img