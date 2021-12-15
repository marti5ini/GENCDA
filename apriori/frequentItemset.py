import os
import re
from collections import defaultdict

import pandas as pd
from fim import apriori
from sklearn.preprocessing import KBinsDiscretizer


class Apriori:
    """
    A class responsible for applying on continuous-valued data the apriori algorithm
    implemented by Christian Borgelt.

    Data preprocessing: we provide a private method to partition continuous features
    into discrete values using KBinsDiscretizer (sklearn).

    References
    ----------
    https://borgelt.net/apriori.html
    https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.KBinsDiscretizer.html?highlight=kbins#sklearn.preprocessing.KBinsDiscretizer

    """

    def __init__(self, dataframe, nbins=4, strategy='quantile', target='m', zmax=3, support=5):
        """
        :param dataframe: pandas Dataframe, continuous variables
        :param nbins: integer, default = 4
            the number of bins to produce
        :param strategy: string, default = 'quantile'
            strategy used to define the widths of the bins.
        :param target: string, default = 'm'
             a character string indicating the type of association mined
        :param support: int, default = 5
            minimum support threshold (percentage of transactions)
        """

        self.df = dataframe
        self.n_bins = nbins
        self.strategy = strategy
        self.target = target
        self.zmax = zmax
        self.support = support
        self.results = set()

    def _baskets(self, discretized_df):

        baskets = defaultdict(list)
        column_names = list(self.df.columns.str.replace('_', ''))
        for i, row in enumerate(discretized_df):
            for elem in range(len(row)):
                baskets[i].append(str(row[elem]) + "_" + str(column_names[elem]))

        basket_list = [b for b in baskets.values()]

        return basket_list

    def _discretizer(self):

        discretizer = KBinsDiscretizer(n_bins=self.n_bins, encode='ordinal', strategy=self.strategy)
        discretizer.fit(self.df)
        discretized_df = discretizer.transform(self.df)

        return discretized_df

    def fit(self):

        itemsets_with_max_support = set()
        discretized_df = self._discretizer()
        itemsets = apriori(self._baskets(discretized_df), supp=self.support, zmax=self.zmax,
                           target=self.target, report='as')
        ordered_itemsets = sorted(itemsets, key=lambda tup: tup[1], reverse=True)

        if len(ordered_itemsets) != 0:
            for pattern in ordered_itemsets:
                matching = re.findall(r"(_)([A-Za-z0-9]*)", str(pattern[0]))
                attr_names = [match[1] for match in matching]
                itemset = frozenset(attr_names)
                itemsets_with_max_support.add(itemset)

        self.results = itemsets_with_max_support

        return self.results

    def get_results(self, to_tuple=False):
        """
        :param to_tuple: bool, default=False
        If it is True, the function returns list of tuples ex. [('0', '1'), ('1', '2')]
        otherwise list of frozenset [frozenset{(('0', '1')}, frozenset{(('1', '2')}]
        :return:
        """

        if self.results is None:
            print("Before get results, use fit method")

        if to_tuple:
            return [tuple(itemset) for itemset in self.results]
        else:
            return self.results


if __name__ == '__main__':
    directory = os.path.realpath(os.path.dirname(__file__))
    file_path = os.path.join(os.path.dirname(directory), "datasets", "synthetic.csv")
    df = pd.read_csv(file_path)
    fim = Apriori(df, zmax=3)
    fim.fit()
    fim.get_results()
