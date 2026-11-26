import numpy as np
from scipy import stats


def cohens_d(a, b):
    """(mean(a) - mean(b)) / pooled sample std. Uses ddof=1 (sample std)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    pooled = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return (a.mean() - b.mean()) / pooled


def welch_p(a, b):
    return float(stats.ttest_ind(a, b, equal_var=False).pvalue)


def magnitude(d):
    d = abs(d)
    return "negligible" if d < 0.2 else "small" if d < 0.5 else "medium" if d < 0.8 else "large"
