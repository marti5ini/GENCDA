from operator import itemgetter

import networkx as nx
import numpy as np
from sklearn.preprocessing import StandardScaler

from apriori.frequentItemset import *
from causal_discovery.hoyer import NonlinearANM
from utils import check_maximal, check_relations_founded, evaluation_measures, check_causal_direction, AutoVivification, \
    delete_cycle


def evaluate(causal_relations, edges, graph, anm_method=True):
    """
    :param causal_relations: tuple list
    :param edges: tuple list
    :param graph: networkx.DiGraph
    :param anm_method: bool, default = False
    :return: dict
    """
    if len(causal_relations) != 0:
        confusion_matrix = check_causal_direction(causal_relations, edges, graph, anm_method)
        return evaluation_measures(confusion_matrix)
    else:
        raise ValueError('Must first run the fitNCD method. If you haves already run the method, it '
                         'has not found any causal relationship.')


class NCDApriori:
    """
    Implementation of NCDA algorithm.
    """

    def __init__(self, dataset):
        """
        :param dataset: pandas.DataFrame

        :param folder_path: string
            The location of the directory to save results
        """
        self.df = dataset
        self.relations = None

    def fitApriori(self, target='m', zmax=3, nbins=4, strategy='quantile', support=5):
        """
        :param target: string, default = 'm'
            a character string indicating the type of association mined
            target type:
            s: frequent item sets
            c: closed (frequent) item sets
            m: maximal (frequent) item sets
        :param zmax: int, default = 3
            Maximum number of items per item set
        :param nbins: integer, default = 4
            the number of bins to produce
        :param strategy: string, default = 'quantile'
            strategy used to define the widths of the bins
        :param support: int, default = 5
            minimum support threshold (percentage of transactions)

        :return: set of frequent item sets
        """
        # Check if all columns are strings
        if not self.df.columns.astype(str).all():
            self.df.columns = self.df.columns.astype(str)
        self.df.columns = self.df.columns.str.replace(' ', '')

        # Instantiate Apriori
        fim = Apriori(self.df, support=support, nbins=nbins, target=target,
                      zmax=zmax, strategy=strategy)
        # Fit Apriori
        fim.fit()

        try:
            if target != 'm':
                fim_results = fim.get_results()
                self.relations = fim_results
            else:
                maximals = fim.get_results(to_tuple=True)
                fim_results = check_maximal(maximals)
                maximals = [frozenset(maximal) for maximal in fim_results]
                self.relations = maximals

            return fim_results

        except IndexError:
            return 'IndexError: Apriori does not detect any frequent itemsets. ' \
                   'Try a different number of bins.'

    def evaluateRelations(self, graph):
        """

        :param graph: networkx.DiGraph
        :return: dict
        """
        edges = [(str(source), str(destination)) for source, destination in graph.edges]
        confusion_matrix = check_relations_founded(self.relations, edges, graph, itemset=True)
        precision, recall, accuracy, f1 = evaluation_measures(confusion_matrix)
        return {'precision': precision, 'recall': recall, 'accuracy': accuracy, 'f1': f1}

    def fitNCD(self, itemsets, alpha=0.001, sorting=np.mean, train_size=0.7, standardization=True):
        """
        :param itemsets:
            set of frequent item sets founded by fitApriori
        :param alpha: float, default = 0.001
            independence test threshold
        :param sorting: function, default = np.mean
            sorting type: [np.mean, np.max, np.min]
            To the aim of returning only the most representative ones,
            we consider only the causal relationships with the highest
            average (or maximum or minimum) level of p-values of the various dependencies.
        :param train_size: float, default = 0.7
            It should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the train split.
        :param standardization: bool, default = True
            standardize features before passing them to
            Gaussian Process Regressor.
        :return:
        """
        if itemsets is not None:
            check_loop = AutoVivification()
            causal_relations = []
            for columns_names in itemsets:
                subset = self.df[list(columns_names)]
                if standardization:
                    # Standardize features by removing the mean and scaling to unit variance
                    scaler = StandardScaler()
                    # Fit to data, then transform it
                    scaled = scaler.fit_transform(subset.values)
                    scaled_df = pd.DataFrame(scaled, columns=list(columns_names))
                    # Instantiate causal discovery method on selected features
                    values = NonlinearANM(scaled_df)
                    # Fit the model
                    possible_dags = values.fit_multivariate(alpha, train_size, sorting)
                else:
                    values = NonlinearANM(subset)
                    possible_dags = values.fit_multivariate(alpha, train_size, sorting)

                # Save causal dependencies found
                dependencies_found = []
                if possible_dags:
                    # Sort results on descending order of p_value
                    prob_dag = sorted(possible_dags, key=itemgetter(1), reverse=True)
                    # Get only dag with max p_value
                    for idx, adjacency_matrix in enumerate([graph for graph, _ in prob_dag]):
                        if idx == 0:
                            for vert, cols in enumerate(adjacency_matrix.T):
                                parents = np.where(cols == 1)[0]
                                if len(parents) > 0:
                                    nodes = [list(columns_names)[par] for par in parents]
                                    parents_list = ', '.join(nodes)
                                    dependencies_found.append((parents_list, list(columns_names)[vert]))

                    if len(dependencies_found) != 0:
                        for dependency in dependencies_found:
                            # Check if the dependency is already included
                            if dependency in causal_relations:
                                continue
                            # If not, add it to a list
                            causal_relations.append(dependency)

                            if dependency in check_loop or re.match(r'\d*\w*,', dependency[0]) is not None:
                                continue
                            check_loop[dependency] = prob_dag[0][1]

            detect_loop = nx.DiGraph(list(check_loop.keys()))
            no_cycle = delete_cycle(check_loop, detect_loop)
            return no_cycle
        else:
            raise ValueError('Must first run the fitApriori method.')


if __name__ == '__main__':
    """path = '/Users/martina/Desktop'
    directory = os.path.realpath(os.path.dirname(__file__))
    file_path = os.path.join(os.path.dirname(directory), "datasets", "synthetic.csv")
    data = pd.read_csv(file_path)
    graph = nx.DiGraph()
    graph.add_nodes_from(['w', 'y', 'x', 'z'])
    graph.add_edges_from([('w', 'x'), ('w', 'y'), ('x', 'z'), ('y', 'z')])
    ncda = NCDApriori(data)
    ncda.fitApriori()"""
    data = pd.read_csv('/Users/martina/Downloads/mice_protein.csv', index_col=False)
    #data.columns = data.columns.str.replace('_', '')
    #data.columns = [x.lower() for x in data.columns]

    ncda = NCDApriori(data.iloc[:, :-1])
    itemsets = ncda.fitApriori(target='m', zmax=3, nbins=4, strategy='quantile', support=5)
    causal_relations = ncda.fitNCD(itemsets, alpha=0.001, sorting=np.mean, train_size=0.7, standardization=True)

    print(causal_relations)