# Distribution-shift tests. Each function takes a reference sample (e.g.,
# training data) and a new sample (e.g., this weeks production wafers) and
# returns a single number quantifying how different they are.
#
# These run per-sensor: a single sensor drifting is a real signal, even if
# the wafer-level scores look fine. In production these would run on a
# rolling window — every day, compare yesterday vs the training reference.

import numpy as np
from scipy import stats


def psi(reference, new, n_bins=10):
    """Population Stability Index between two 1D samples.

    Bins the reference distribution into n_bins quantiles, then measures
    how much probability mass shifted between bins. Convention:
        < 0.1   stable
        0.1-0.25  moderate shift, monitor
        > 0.25  significant shift, investigate

    We add a tiny epsilon to empty bins so the log doesnt blow up; this is
    the standard trick and matches how the credit-risk libraries implement it.
    """
    eps = 1e-6

    # Bin edges from the reference sample. quantile edges give equal-mass
    # bins under the reference, which makes PSI more sensitive to shape
    # changes than equal-width bins would.
    edges = np.quantile(reference, np.linspace(0, 1, n_bins + 1))
    # Slightly extend the outermost edges so points exactly on the boundary
    # land in a bin.
    edges[0]  -= 1e-9
    edges[-1] += 1e-9

    ref_counts, _ = np.histogram(reference, bins=edges)
    new_counts, _ = np.histogram(new,       bins=edges)

    ref_frac = ref_counts / max(ref_counts.sum(), 1)
    new_frac = new_counts / max(new_counts.sum(), 1)

    ref_frac = np.where(ref_frac == 0, eps, ref_frac)
    new_frac = np.where(new_frac == 0, eps, new_frac)

    return float(np.sum((new_frac - ref_frac) * np.log(new_frac / ref_frac)))


def ks_statistic(reference, new):
    """Two-sample Kolmogorov-Smirnov statistic and p-value.

    Returns (statistic, p_value). Statistic is in [0, 1]: 0 means
    identical CDFs, 1 means fully separated. p-value below 0.05 means
    "the two samples likely came from different distributions".
    """
    result = stats.ks_2samp(reference, new)
    return float(result.statistic), float(result.pvalue)


def mmd_rbf(reference, new, gamma=None):
    """Squared MMD between two samples under an RBF kernel.

    Unlike PSI and KS this can operate on multivariate inputs — pass an
    (n, d) array and it picks up shifts in the joint distribution that
    per-sensor tests would miss. We use the unbiased estimator. gamma=None
    uses the median heuristic (a robust default for kernel bandwidth).

    Returns a single non-negative number. Larger = more drift. Theres no
    universal threshold for MMD — you compare against a permutation null
    or against historical values to decide what counts as anomalous.
    """
    ref = np.atleast_2d(reference)
    new = np.atleast_2d(new)
    if ref.shape[0] == 1: ref = ref.T
    if new.shape[0] == 1: new = new.T

    if gamma is None:
        # Median heuristic: bandwidth set to the median squared distance
        # between points in the combined sample. Well-known robust default.
        combined = np.vstack([ref, new])
        # Sample for speed if the combined set is large
        sample_size = min(500, combined.shape[0])
        idx = np.random.RandomState(0).choice(combined.shape[0], sample_size, replace=False)
        sub = combined[idx]
        diffs = sub[:, None, :] - sub[None, :, :]
        sq_dists = (diffs ** 2).sum(axis=-1)
        median_sq = np.median(sq_dists[sq_dists > 0])
        gamma = 1.0 / (median_sq + 1e-12)

    def rbf(a, b):
        diffs = a[:, None, :] - b[None, :, :]
        return np.exp(-gamma * (diffs ** 2).sum(axis=-1))

    k_rr = rbf(ref, ref)
    k_nn = rbf(new, new)
    k_rn = rbf(ref, new)

    # Unbiased MMD^2: exclude the diagonal terms in the within-sample sums
    n_r, n_n = ref.shape[0], new.shape[0]
    sum_rr = (k_rr.sum() - np.trace(k_rr)) / (n_r * (n_r - 1))
    sum_nn = (k_nn.sum() - np.trace(k_nn)) / (n_n * (n_n - 1))
    sum_rn = k_rn.mean()

    return float(sum_rr + sum_nn - 2 * sum_rn)
