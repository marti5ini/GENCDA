import numpy as np
import scipy
import scipy.stats
import matplotlib.pyplot as plt
import warnings


class Distribution(object):

    def __init__(self, dist_names_list=[]):
        self.dist_names = ['norm', 'lognorm', 'uniform', 'exponweib', 'expon',
                           'gamma', 'beta', 'alpha', 'chi', 'chi2', 'laplace', 'powerlaw']
        self.dist_results = []
        self.params = {}

        self.DistributionName = ""
        self.PValue = 0
        self.Param = None

        self.isFitted = False

    def freedman_diaconis(self, x):
        iqr = np.subtract(*np.percentile(x, [75, 25]))
        n = len(x)
        h = max(2.0 * iqr / n ** (1.0 / 3.0), 1)
        k = np.ceil((np.max(x) - np.min(x)) / h)
        return k

    def struges(self, x):
        n = len(x)
        k = np.ceil(np.log2(n)) + 1
        return k

    def estimate_nbr_bins(self, x):
        if len(x) == 1:
            return 1
        k_fd = self.freedman_diaconis(x) if len(x) > 2 else 1
        k_struges = self.struges(x)
        if k_fd == float('inf') or np.isnan(k_fd):
            k_fd = np.sqrt(len(x))
        k = max(k_fd, k_struges)
        return k

    def Fit(self, y):
        self.dist_results = []
        self.params = {}

        for dist_name in self.dist_names:
            dist = getattr(scipy.stats, dist_name)
            # Try to fit the distribution
            try:
                # Ignore warnings from data that can't be fit
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore')

                    param = dist.fit(y)
                    self.params[dist_name] = param
                    # Applying the Kolmogorov-Smirnov test
                    D, p = scipy.stats.kstest(y, dist_name, args=param)
                    self.dist_results.append((dist_name, p))

            except Exception:
                pass

        # select the best fitted distribution
        sel_dist, p = (max(self.dist_results, key=lambda item: item[1]))
        # store the name of the best fit and its p value
        self.DistributionName = sel_dist
        self.PValue = p

        self.isFitted = True
        return self.DistributionName, self.PValue

    def Fit_bins(self, data):
        self.dist_results = []
        self.params = {}
        bins = int(np.round(self.estimate_nbr_bins(data)))
        y, x = np.histogram(data, bins=bins, density=True)
        x = (x + np.roll(x, -1))[:-1] / 2.0

        for dist_name in self.dist_names:
            dist = getattr(scipy.stats, dist_name)
            param = dist.fit(x)

            self.params[dist_name] = param
            # Applying the Kolmogorov-Smirnov test
            D, p = scipy.stats.kstest(y, dist_name, args=param);
            self.dist_results.append((dist_name, p))

        # select the best fitted distribution
        sel_dist, p = (max(self.dist_results, key=lambda item: item[1]))
        # store the name of the best fit and its p value
        self.DistributionName = sel_dist
        self.PValue = p

        self.isFitted = True
        return self.DistributionName, self.PValue

    def Random(self, n=1000):
        if self.isFitted:
            dist_name = self.DistributionName
            param = self.params[dist_name]
            # initiate the scipy distribution
            dist = getattr(scipy.stats, dist_name)
            return dist.rvs(*param[:-2], loc=param[-2], scale=param[-1], size=n)
        else:
            raise ValueError('Must first run the Fit method.')

    def Plot(self, data, x=None):
        if x is None:
            x = self.Random(n=len(data))

        plt.hist(data, alpha=0.5, label='Actual')
        plt.hist(x, alpha=0.5, label='Fitted')
        plt.legend(loc='upper right')