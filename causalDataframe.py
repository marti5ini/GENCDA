import pandas as pd
import random

from utils import *

UNARY_FUNCTIONS = (np.sin, sqrt, square, log, np.tan, np.cos)
BINARY_FUNCTIONS = (sum, product, difference, division)
DISTRIBUTIONS = (np.random.uniform, np.random.normal, np.random.exponential, np.random.lognormal, np.random.chisquare,
                 np.random.beta)


class CausalDataFrame:
    """
    Generate multivariate data with dependency causal structure specified by a given DAG (Directed Acyclic Graph)
    with nodes corresponding to variables and edges representing causal relationships.
    """

    def __init__(self, n_points, graph_data, isolated_nodes=[],
                 possible_unary_functions=UNARY_FUNCTIONS,
                 possible_binary_functions=BINARY_FUNCTIONS, distributions=DISTRIBUTIONS):
        """
        :param n_points: integer
            the number of samples
        :param graph_data: a graph object describing the DAG
            It could be: any NetworkX graph, dict-of-dicts, dict-of-lists,
            container (e.g. set, list, tuple) of edges, iterator (e.g. itertools.chain) that produces edges,
            generator of edges, Pandas DataFrame (row per edge)
            numpy matrix, numpy ndarray, scipy sparse matrix, pygraphviz agraph
        :param isolated_nodes: list, default = []
             vertex with degree zero.
             If you initialize "graph_data" using only edges, the networkx function generates
             exclusively connected components. In this case, you have to insert the list of isolated nodes
             to add them as variables in the pandas dataframe.
        :param possible_unary_functions: tuple, default = (np.sin, sqrt, square, log, np.tan, np.cos)
            list of possible unary function to generate dependent column
        :param possible_binary_functions: tuple, default = (sum, product, difference, division)
            list of possibile binary function to generate dependent column
        :param distributions: tuple, default = (np.random.uniform, np.random.normal, np.random.exponential,
                                                np.random.lognormal, np.random.chisquare, np.random.beta)
            list of distribution to generate independent column
        """

        self.n_sample = n_points
        self.possible_unary_functions = possible_unary_functions
        self.possible_binary_functions = possible_binary_functions
        self.distributions = distributions
        g = nx.to_networkx_graph(graph_data, create_using=nx.DiGraph)

        g.add_nodes_from(isolated_nodes)

        self.graph = g
        self.n_features = len(self.graph.nodes)

        if not nx.is_directed_acyclic_graph(self.graph):
            raise ValueError('The graph contains a cycle.')

        self._values = {}
        self.dataframe = None

    def _get_distribution(self):
        """
        :return: random samples of randomly chosen distribution
        """
        attr_uniform = np.random.randint(5, 100)
        scale_uniform = -attr_uniform, attr_uniform
        scale_normal = np.random.randint(0, 100), np.random.randint(1, 5)
        scale_exponential = (np.random.randint(1, 50))
        scale_log = np.random.randint(0, 5), np.random.randint(5, 10)
        scale_chi = (np.random.randint(1, 10))
        scale_beta = round(random.uniform(0.1, 0.8), 2), round(random.uniform(0.9, 2.0), 2)
        parameters = (scale_uniform, scale_normal, scale_exponential, scale_log, scale_chi, scale_beta)
        name, scale = random.choice(list(zip(self.distributions, parameters)))
        if isinstance(scale, tuple):
            column = name(scale[0], scale[1], self.n_sample)
        else:
            column = name(scale, self.n_sample)
        return column

    def _generate_column(self, node, independent_col_distribution, parents_list=None):
        if len(parents_list) == 0:
            if independent_col_distribution == 'uniform':
                attr_scale = random.randint(5, 100)
                column = np.random.uniform(0, attr_scale, self.n_sample)
            else:
                column = self._get_distribution()
            noise_scale = 1
            noise = np.random.uniform(-noise_scale, noise_scale, self.n_sample)
        else:
            column = None
            for parent in parents_list:
                parent_column = self._values[parent]
                function_idx = random.randint(0, len(self.possible_unary_functions) - 1)
                tmp_parent_column = self.possible_unary_functions[function_idx](parent_column)
                if column is None:
                    column = tmp_parent_column
                else:
                    arithmetic_operator_idx = random.randint(0, len(self.possible_binary_functions) - 1)
                    column = self.possible_binary_functions[arithmetic_operator_idx](column, tmp_parent_column)
            noise_scale = 1
            noise = np.random.uniform(-noise_scale, noise_scale, self.n_sample)

        self._values[node] = column + noise

        return

    def generate_data(self, independent_col_distribution='uniform'):
        """
        :param independent_col_distribution: str, default = 'uniform'
            Distribution of independent features. Two possible values: default or 'random'.
            1) independent features are uniformly distributed on the interval [0, rand].
            rand is a random integer between 5 and 100.
            2) independent features distribution is chosen random from a list of numpy distribution
            with scale = random integer, size = n_of_samples
            Either way, it is added noise uniformly distributed on the interval [-1, 1].
        """
        valid_status = {'uniform', 'random'}
        if independent_col_distribution not in valid_status:
            raise ValueError("independent_col_distribution: status must be one of %r." % valid_status)

        for node in nx.topological_sort(self.graph):
            predecessors = list(self.graph.predecessors(node))
            self._generate_column(node, independent_col_distribution, predecessors)

        self.dataframe = pd.DataFrame(self._values).sort_index(axis=1)

        return

    def get_edges(self, edge_type='str'):
        """
        :param edge_type: str, default = 'str'
            Type of edge in the graph.
        :return: list of edges
        """
        valid_status = {'str', 'int'}
        if edge_type not in valid_status:
            raise ValueError("edge type: status must be one of %r." % valid_status)
        if edge_type == 'int':
            return [e for e in self.graph.edges]
        else:
            return [(str(source), str(destination)) for source, destination in self.graph.edges]

    def show_graph(self):
        """
        :return:  Draw the graph using networkx
        """
        nx.draw_networkx(self.graph, node_size=1500, font_color='w', font_size=16)
        plt.show()
        return


if __name__ == '__main__':
    edges = [('0', '2'), ('1', '2'), ('2', '7'), ('1', '4'), ('4', '8')]
    d = CausalDataFrame(1500, edges, ['3', '5', '6'])
    d.generate_data()
    df = d.dataframe
    df.to_csv("/Users/martina/Documents/df.csv", index=False)
    print(df[0:5])
    d.show_graph()
    plt.savefig('/Users/martina/Documents/ground_truth.png')
    plt.show()
