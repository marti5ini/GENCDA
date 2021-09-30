import pandas as pd
import networkx as nx
from data_generation.fitDistribution import *
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor


class RelatedDataframe:

    def __init__(self, df, dag, ensemble_regression=[DecisionTreeRegressor(), SVR(),
                                                     KNeighborsRegressor(), GaussianProcessRegressor()]):
        self.df = df
        self.n_sample = len(df)
        self.graph = dag
        self._values = {}
        self.dataframe = None
        self.ensemble_regression = ensemble_regression

    def regressor(self, independent_cols, dependent_col):

        regression_result = None

        for model in self.ensemble_regression:
            model.fit(independent_cols, dependent_col.reshape(-1, 1))
            y_pred = model.predict(independent_cols).reshape(-1, 1)

            if regression_result is None:
                regression_result = y_pred
            else:
                column = y_pred
                regression_result = np.concatenate((regression_result, column),
                                                   axis=1)

        return np.mean(regression_result, axis=1)

    def _generate_column(self, node, parents_list=None):

        if len(parents_list) == 0:
            feature_scaled = StandardScaler()
            scaled = feature_scaled.fit_transform(self.df[node].values.reshape(-1, 1))
            dst = Distribution()
            dst.Fit(self.df[node])
            column = dst.Random(self.n_sample)
            #dst.Plot(self.df[node])
            # plt.show()
            column = feature_scaled.inverse_transform(column)

        else:
            column = self.regressor(self.df[parents_list].values, self.df[node].values)
            #plt.hist(self.df[node], alpha=0.5, label='Actual')
            #plt.hist(column, alpha=0.5, label='Fitted')
            #plt.legend(loc='upper right')
            #plt.show()

        self._values[node] = column

    def generate_data(self, is_sorted=True):
        """

        :param is_sorted: bool, default = True
            Choose if order index
        :return: pandas DataFrame
        """
        for node in nx.topological_sort(self.graph):
            predecessors = list(self.graph.predecessors(node))
            self._generate_column(node, predecessors)

        if is_sorted:
            self.dataframe = pd.DataFrame(self._values).sort_index(axis=1)
        else:
            self.dataframe = pd.DataFrame(self._values)

        return self.dataframe

