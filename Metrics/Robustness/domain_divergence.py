import numpy as np
from sklearn.metrics.pairwise import rbf_kernel, euclidean_distances

def compute_mmd_rbf(X_source, X_target, gamma=None):
    """
    Computes Maximum Mean Discrepancy (MMD) with RBF kernel between two datasets.

    MMD²(P, Q) = E_{x,x'∼P}[k(x,x')] + E_{y,y'∼Q}[k(y,y')] - 2 * E_{x∼P,y∼Q}[k(x,y)]
    where k(x, y) = exp(-γ ||x - y||²) is the RBF kernel.

    Parameters:
    - X_source: np.ndarray, source domain data (n_samples_src x n_features)
    - X_target: np.ndarray, target domain data (n_samples_tgt x n_features)
    - gamma: float, RBF kernel bandwidth parameter. If None, uses median heuristic.

    Returns:
    - mmd: float, the estimated MMD² between domains.

    Reference:
    Gretton, A., Borgwardt, K. M., Rasch, M. J., Schölkopf, B., & Smola, A. J. (2012).
    "A Kernel Two-Sample Test". Journal of Machine Learning Research, 13, 723–773.
    https://www.jmlr.org/papers/volume13/gretton12a/gretton12a.pdf

    Interpretation:
    - MMD^2 = 0 → identical distributions
    - Higher values indicate greater divergence between source and target domains
    """

    # Ensure input is numpy array
    X_source = np.asarray(X_source)
    X_target = np.asarray(X_target)

    # Compute gamma using median heuristic if not provided
    if gamma is None:
        pairwise_dists = euclidean_distances(X_source, X_target)
        median_dist = np.median(pairwise_dists)
        gamma = 1.0 / (2 * median_dist**2)
        print(f"[Info] Gamma estimated via median heuristic: {gamma:.4f}")

    # Compute RBF kernels
    K_ss = rbf_kernel(X_source, X_source, gamma=gamma)
    K_tt = rbf_kernel(X_target, X_target, gamma=gamma)
    K_st = rbf_kernel(X_source, X_target, gamma=gamma)

    # Compute MMD
    m = X_source.shape[0]
    n = X_target.shape[0]
    mmd = K_ss.mean() + K_tt.mean() - 2 * K_st.mean()

    return mmd
