# [1] D'Agostino, R. B., Stephens, M. A., Goodness-of-fit-techniques.
#     Routledge, 2017.

import math
import warnings

import numpy as np

from scipy.stats import anderson, ks_1samp, norm, normaltest, shapiro


def anderson_darling(x: np.ndarray) -> float:
    """Anderson-Darling test for normality.

    Args:
        x: Input 1D sample, shape (n,)

    Returns:
        p-value.
    """
    # We only need the test statistic; the p-value is computed manually
    # from reference tables below.  Silence the SciPy >=1.17 FutureWarning
    # that asks callers to choose a p-value method.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning,
                                message=".*method.*")
        A2 = anderson(x, dist='norm').statistic
    n = len(x)
    # Modified statistic [1, Table 4.7]
    A2 = A2*(1 + (.75/n) + 2.25/(n**2))
    # p-value [1, Table 4.9]
    if A2 >= .6:
        p = math.exp(1.2937 - 5.709*A2 - .0186*(A2**2))
    elif A2 >= .34:
        p = math.exp(.9177 - 4.279*A2 - 1.38*(A2**2))
    elif A2 > .2:
        p = 1 - math.exp(-8.318 + 42.796*A2 - 59.938*(A2**2))
    else:
        p = 1 - math.exp(-13.436 + 101.14*A2 - 223.73*(A2**2))
    return p


# input: (n_sample,) array
# output: float, p-value
TESTS = {
    'ad': anderson_darling,                        # Anderson-Darling
    'dp': lambda x: normaltest(x).pvalue,          # D'Agostino-Pearson
    'ks': lambda x: ks_1samp(x, norm.cdf).pvalue,  # Kolmogorov-Smirnov
    'sw': lambda x: shapiro(x).pvalue              # Shapiro-Wilk
}