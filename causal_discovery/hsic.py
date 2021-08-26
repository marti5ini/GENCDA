"""
    Implementation of Hilbert Schmidt Independence Criterion

    hsic_gam implements the HSIC test using a Gamma approximation.

    Gretton, A., Fukumizu, K., Teo, C. H., Song, L., Scholkopf, B.,
    & Smola, A. J. (2007). A kernel statistical test of independence.
    In Advances in neural information processing systems (pp. 585-592).

    Based on python code of Shoubo - https://github.com/amber0309/HSIC/blob/master/HSIC.py

    Null hypothesis: independent variables

    Inputs:
    X 		numpy vector with row - sample, col - dim_X
    Y 		numpy vector with row - sample, col - dim_Y

    Outputs:
    p_value         significance of result

"""

import numpy as np
from scipy.stats import gamma


def rbf_dot(pattern1, pattern2, deg):
    size1 = pattern1.shape
    size2 = pattern2.shape

    G = np.sum(pattern1 * pattern1, 1).reshape(size1[0], 1)
    H = np.sum(pattern2 * pattern2, 1).reshape(size2[0], 1)

    Q = np.tile(G, (1, size2[0]))
    R = np.tile(H.T, (size1[0], 1))

    H = Q + R - 2 * np.dot(pattern1, pattern2.T)

    H = np.exp(-H / 2 / (deg ** 2))

    return H


def hsic_gam(X, Y):
    """
    :param X: numpy vector with row - sample, col - dim
    :param Y: numpy vector with row - sample, col - dim
    :return: integer
    """
    n = X.shape[0]

    Xmed = X

    G = np.sum(Xmed * Xmed, 1).reshape(n, 1)
    Q = np.tile(G, (1, n))
    R = np.tile(G.T, (n, 1))

    dists = Q + R - 2 * np.dot(Xmed, Xmed.T)
    dists = dists - np.tril(dists)
    dists = dists.reshape(n ** 2, 1)

    width_x = np.sqrt(0.5 * np.median(dists[dists > 0]))

    Ymed = Y

    G = np.sum(Ymed * Ymed, 1).reshape(n, 1)
    Q = np.tile(G, (1, n))
    R = np.tile(G.T, (n, 1))

    dists = Q + R - 2 * np.dot(Ymed, Ymed.T)
    dists = dists - np.tril(dists)
    dists = dists.reshape(n ** 2, 1)

    width_y = np.sqrt(0.5 * np.median(dists[dists > 0]))

    bone = np.ones((n, 1), dtype=float)
    H = np.identity(n) - np.ones((n, n), dtype=float) / n

    K = rbf_dot(X, X, width_x)
    L = rbf_dot(Y, Y, width_y)

    Kc = np.dot(np.dot(H, K), H)
    Lc = np.dot(np.dot(H, L), H)

    testStat = np.sum(Kc.T * Lc) / n

    varHSIC = (Kc * Lc / 6) ** 2

    varHSIC = (np.sum(varHSIC) - np.trace(varHSIC)) / n / (n - 1)

    varHSIC = varHSIC * 72 * (n - 4) * (n - 5) / n / (n - 1) / (n - 2) / (n - 3)

    K = K - np.diag(np.diag(K))
    L = L - np.diag(np.diag(L))

    muX = np.dot(np.dot(bone.T, K), bone) / n / (n - 1)
    muY = np.dot(np.dot(bone.T, L), bone) / n / (n - 1)

    mHSIC = (1 + muX * muY - muX - muY) / n

    al = mHSIC ** 2 / varHSIC
    bet = varHSIC * n / mHSIC

    p_value = float(1 - gamma(al, scale=bet).cdf(testStat))

    return p_value
