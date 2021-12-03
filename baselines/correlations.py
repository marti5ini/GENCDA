from itertools import combinations
import networkx as nx

from scipy.stats import spearmanr, pearsonr

from baselines.XtendendCorr import *
from utils import check_relations_founded


class Correlation(object):

    def __init__(self):
        super(Correlation, self).__init__()
        self.metrics = None

    @staticmethod
    def pearson(column1, column2):
        """
        Compute test statistic
        :param column1, numpy array
        :param column2, numpy array
        :return: r and p-value
        """
        return pearsonr(column1, column2)

    @staticmethod
    def spearman(column1, column2):
        """
        Compute test statistic
        :param column1, numpy array
        :param column2, numpy array
        :return: r and p-value
        """
        return spearmanr(column1, column2)

    @staticmethod
    def hoeffding(column1, column2):
        """
        Compute test statistic
        :param column1, numpy array
        :param column2, numpy array
        :return: r
        """
        return hoeffding(column1, column2)

    def pairwise(self, df, method=pearsonr):
        """
        Compute pairwise correlation of columns
        :param df: pandas DataFrame
        :param method: function of the correlation method, default = pearson
            {'pearsonr', 'spearmanr', 'hoeffding'}
        :return: DataFrame
        """
        self.metrics = method
        correlations = {}
        # Check if all columns are strings
        if df.columns.dtype != 'str':
            df.columns = df.columns.astype(str)

        columns = list(df.columns)
        for col_a, col_b in combinations(columns, 2):
            correlations[col_a + '-' + col_b] = method(df.loc[:, col_a].values, df.loc[:, col_b].values)

        if method == hoeffding:
            return pd.DataFrame(correlations, index=[0])
        else:
            return pd.DataFrame.from_dict(correlations)

    def evaluate(self, df, graph, p_value=0.05, thresold=0.03):
        """
        Compute and evaluate these correlation metrics: pearson, spearman, hoeffding.
        Evaluation of pearson and spearman: p_value
                   of hoeffding: threshold of 0.03

        Hoeffding's D lies on the interval [-.5, 1] if there are no tied ranks, with larger
        values indicating a stronger relationship between the variables.
        (https://m-clark.github.io/docs/CorrelationComparison.pdf)

        :param df: pandas DataFrame
        :param graph: networkx.DiGraph
        :param p_value: default = 0.05
        :param thresold: test threshold for Hoeffding's D
        :return: correlations list, confusion matrix of correlations founded wrt original graph


        """
        edges = [(str(source), str(destination)) for source, destination in graph.edges]

        if self.metrics is None:
            raise ValueError('You have to run pairwise function before.')

        result = []

        for column in range(len(df.columns)):
            if self.metrics is pearsonr or self.metrics is spearmanr:
                p_value_to_check = df.iloc[1, column]
                if p_value_to_check < p_value:
                    columns_names = df.iloc[:, column].name
                    result.append(columns_names)
            else:
                hoeff_corr = df.iloc[0, column]
                if hoeff_corr >= thresold:
                    columns_names = df.iloc[:, column].name
                    result.append(columns_names)

        # If the method find relations
        if len(result) != 0:
            # We evaluate them with undirected acyclic graph
            matrix = check_relations_founded(result, edges, graph)
            return result, matrix
        else:
            print('No correlations found.')


if __name__ == '__main__':
    from sklearn import datasets
    import matplotlib.pyplot as plt

    diabetes = datasets.load_diabetes()
    dataframe = diabetes['data'][:, [0, 1, 2, 3, 4]]
    data = pd.DataFrame(dataframe, columns=['10', '11', '12', '13', '14'])
    #pearson = Correlation().pearson(data.iloc[:, 0], data.iloc[:, 1])

    graph = nx.DiGraph()
    graph.add_nodes_from([str(i) for i in range(10, 15)])
    graph.add_edges_from([('0', '1')])
    nx.draw_networkx(graph)
    plt.show()
    # main class
    corr = Correlation()
    # Compute test statistic between two variables
    r, p_value = corr.pearson(data.iloc[:, 0], data.iloc[:, 1])
    # Compute pairwise correlation of columns
    new_df = corr.pairwise(data, pearsonr)
    columns, confusion_matrix = corr.evaluate(new_df, graph)
    print(f'List of indices of column pairs that are correlated: {columns}')

    """
    Evaluation of the number of relations found comparing relations found by correlation metric and the ground truth. 
    In this case, we verify relations on an undirect acyclic graph.

    """
    # precision, recall, accuracy, f1 = evaluation_measures(confusion_matrix)
    # print(f'Precision: {precision}\nRecall: {recall}\nAccuracy: {accuracy}\nF1: {f1}')
