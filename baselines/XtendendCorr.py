import math
import warnings

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.preprocessing import KBinsDiscretizer

warnings.filterwarnings("ignore", category=UserWarning)

"""
Code from https://github.com/PaulVanDev/HoeffdingD
A proper implementation in python is not existing so far. 
The original algorithm presents more complexity (O(n2)) than other popular correlation coefficients like Pearson or Spearman 
and was basically done for the statistics in the second part of the 20th century 
(which means a size order of ~100 points). 
Nowadays, the "big data period", it's common to search relations between time-series with millions of points. 
On could say that it is not often relevant (if there's no stationarity), but the move for automated analytical 
tools plays in favor of such develoment. 
The remark is anyway valid for other correlation coefficients. 
This coefficient is available in some premium analytics sofware (SAS, JMP; in an efficient implementation), 
in R (which is originally more dedicated to statistics), in Matlab (for small data), but unavailable in Python. 
So we propose here an efficient implementation in python3, which is callable as a method by the corr() function 
for dataframe in Pandas.

The development steps were the followings:

Starting algorithm in Matlab
1) Rough adaptation in python
2) Code optimisation - complexity O(n²) (33x faster on 1000 points -> 2,40ms)
3) Constrained algorithm-> binning on entries (50) and resampling if n >100000 – acceptable approximation
4) Support dataFrame as input
5) Compatible with correlation function correlation in Pandas (from v0.24)
"""


def hoeffding(*arg):
    if (len(arg) == 1):
        if isinstance(arg[0], pd.DataFrame):
            if (arg[0].shape[0] > 1):
                return arg[0].apply(lambda x: arg[0].apply(lambda y: hoeffding(x.values, y.values)))
    else:
        if (len(arg) == 2):
            if type(arg[0]) is not np.ndarray:
                if (len(arg[0].shape) > 1):
                    return print("ERROR inputs : hoeffding(df >2col) or hoeffding(numpy.array -1d- ,numpy.array -1d-)")
            if type(arg[1]) is np.ndarray:
                if (len(arg[0].shape) > 1):
                    return print("ERROR inputs : hoeffding(df >2col) or hoeffding(numpy.array -1d- ,numpy.array -1d-)")

            xin = arg[0]
            yin = arg[1]
            # crop data to the smallest array, length have to be equal
            if len(xin) < len(yin):
                yin = yin[:len(xin)]
            if len(xin) > len(yin):
                xin = xin[:len(yin)]

            # dropna
            x = xin[~(np.isnan(xin) | np.isnan(yin))]
            y = yin[~(np.isnan(xin) | np.isnan(yin))]

            # undersampling if length too long
            lenx = len(x)
            if lenx > 99999:
                factor = math.ceil(lenx / 100000)
                x = x[::factor]
                y = y[::factor]

            # bining if too much "definition"
            if len(np.unique(x)) > 50:
                est = KBinsDiscretizer(n_bins=50, encode='ordinal',
                                       strategy='quantile')  # faster strategy='quantile' but less accurate
                est.fit(x.reshape(-1, 1))
                Rtemp = est.transform(x.reshape(-1, 1))
                R = rankdata(Rtemp)
            else:
                R = rankdata(x)
            if len(np.unique(y)) > 50:
                est1 = KBinsDiscretizer(n_bins=50, encode='ordinal',
                                        strategy='quantile')  # faster strategy='quantile' but less accurate
                est1.fit(y.reshape(-1, 1))
                Stemp = est1.transform(y.reshape(-1, 1))
                S = rankdata(Stemp)
            else:
                S = rankdata(y)

                # core processing
            N = x.shape
            dico = {(np.nan, np.nan): np.nan}
            dicoRin = {np.nan: np.nan}
            dicoSin = {np.nan: np.nan}
            dicoRless = {np.nan: np.nan}
            dicoSless = {np.nan: np.nan}
            Q = np.ones(N[0])

            i = 0;
            for r, s in np.nditer([R, S]):
                r = float(r)
                s = float(s)
                if (r, s) in dico.keys():
                    Q[i] = dico[(r, s)]
                else:
                    if r in dicoRin.keys():
                        isinR = dicoRin[r]
                        lessR = dicoRless[r]
                    else:
                        isinR = np.isin(R, r)
                        dicoRin[r] = isinR
                        lessR = np.less(R, r)
                        dicoRless[r] = lessR

                    if s in dicoSin.keys():
                        isinS = dicoSin[s]
                        lessS = dicoSless[s]
                    else:
                        isinS = np.isin(S, s)
                        dicoSin[s] = isinS
                        lessS = np.less(S, s)
                        dicoSless[s] = lessS

                    Q[i] = Q[i] + np.count_nonzero(lessR & lessS) \
                           + 1 / 4 * (np.count_nonzero(isinR & isinS) - 1) \
                           + 1 / 2 * (np.count_nonzero(isinR & lessS)) \
                           + 1 / 2 * (np.count_nonzero(lessR & isinS))
                    dico[(r, s)] = Q[i]
                i += 1

            D1 = np.sum(np.multiply((Q - 1), (Q - 2)));
            D2 = np.sum(np.multiply(np.multiply((R - 1), (R - 2)), np.multiply((S - 1), (S - 2))));
            D3 = np.sum(np.multiply(np.multiply((R - 2), (S - 2)), (Q - 1)));

            D = 30 * ((N[0] - 2) * (N[0] - 3) * D1 + D2 - 2 * (N[0] - 2) * D3) / (
                        N[0] * (N[0] - 1) * (N[0] - 2) * (N[0] - 3) * (N[0] - 4));

            return D
        return print("ERROR inputs : hoeffding(df >2col) or hoeffding(numpy.array -1d- ,numpy.array -1d-)")