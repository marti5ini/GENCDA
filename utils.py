import json
import os
import re
from itertools import combinations
from itertools import product

import matplotlib.pyplot as plt
import networkx as netx
import networkx as nx
import numpy as np
from networkx.readwrite import json_graph


def sqrt(x):
    return np.sqrt(np.abs(x))


def square(x):
    return x * x


def log(x):
    return np.log(np.abs(x))


def sum(x, y):
    return x + y


def difference(x, y):
    return x - y


def division(x, y):
    try:
        z = x / y
    except ZeroDivisionError:
        z = np.zeros_like(x)
    return z


def multiplication(x, y):
    return x * y


def check_maximal(frequent_itemset):
    n = len(frequent_itemset)
    is_maximal = np.ones(n)
    for i in range(n):
        s1 = set(frequent_itemset[i])
        for j in range(n):
            if i == j:
                continue
            s2 = set(frequent_itemset[j])
            if s1 < s2:
                is_maximal[i] = 0
                break

    maximals = []
    for i in range(n):
        if is_maximal[i]:
            maximals.append(frequent_itemset[i])

    return maximals


def is_acyclic(adjacency_matrix):
    """
    adjacency_matrix is acyclic if, and only if, exists the inverse of his identity matrix
    minus the matrix and it his positive.
    """
    try:
        return np.all(np.linalg.inv(np.eye(adjacency_matrix.shape[0]) - adjacency_matrix) >= 0)
    except np.linalg.LinAlgError as e:
        if 'Singular matrix' in str(e):
            return False
        else:
            raise ValueError("The graph contains a cycle.")


def has_parents(adjacency_matrix):
    """
    param adjacency_matrix: numpy ndarray
    return: list of list
    """
    parent = [[] for _ in range(adjacency_matrix.shape[0])]
    for i, col in enumerate(adjacency_matrix.T):
        for elem in range(len(col)):
            if col[elem] == 1:
                parent[i].append(elem)
    return parent


def has_cycle_adjacent_vertices(adjacency_matrix):
    for i, row in enumerate(adjacency_matrix):
        for j, cols in enumerate(row):
            if cols == 1 and adjacency_matrix[j, i] == 1:
                return True
    return False


def all_matrices(n):
    """
    generate all possible binary matrices n*n
    param n: int
    return: list of numpy ndarray
    """
    matrices = []
    for i in product([0, 1], repeat=n * n):
        adj_matrix = np.reshape(np.array(i), (n, n))
        diagonal = adj_matrix.diagonal()
        self_loops = np.count_nonzero(diagonal)
        all_zeros = np.equal(adj_matrix, np.zeros(adj_matrix.shape)).all()
        if self_loops == 0 and not all_zeros and not has_cycle_adjacent_vertices(adj_matrix):
            matrices.append(adj_matrix)
    return matrices


def show_dependency(results, columns, show=True):
    n_dependencies = 0
    for idx, adjacency_matrix in enumerate(results):
        n_dependencies += 1
        if show:
            print(f'Causal dependence n: {idx}')
        for vert, cols in enumerate(adjacency_matrix.T):
            parents = np.where(cols == 1)[0]
            if len(parents) > 0 and show:
                print(f'{[columns[par] for par in parents]} --> {columns[vert]}')
    return n_dependencies


def graph_labels(columns):
    labels = dict()
    for i, column in enumerate(columns):
        labels[i] = column
    return labels


def show_graph(result, labels):
    for matrix in result:
        gr = netx.from_numpy_matrix(matrix, create_using=netx.DiGraph())
        netx.draw_networkx(gr, with_labels=True, labels=labels, node_size=600)
        plt.show()


def check_relations_founded(relations_founded, edges, graph, itemset=False):
    """
    :param relations_founded: frozenset list
    :param edges: tuple list
    :param graph: networkx DiGraph
    :param itemset: bool, default=False
    :return: confusion matrix dict
    """
    edges = set([frozenset(str(edge)) if isinstance(edge, int) else frozenset(edge) for edge in edges])
    if itemset:
        results = set()
        for relation in relations_founded:
            if len(relation) > 2:
                comb = {frozenset(el) for el in combinations(relation, 2)}
                results = results.union(comb)
            else:
                results.add(relation)
    else:
        results = set()
        for causal in relations_founded:
            groups = re.search(r'(\w*)-(\w*)', causal)
            results.add(frozenset(groups.groups()))

    non_edges = set(frozenset(elem) for elem in nx.non_edges(graph.to_undirected()))

    tp = edges.intersection(results)
    tn = non_edges.difference(results)
    fn = edges.difference(results)
    fp = results.difference(edges)

    return {'TP': len(tp), 'FP': len(fp), 'FN': len(fn), 'TN': len(tn)}


def evaluation_measures(confusion_matrix):
    """
    Compute statistics for directed acyclic graphs.
    :param confusion_matrix: dict
        A table with two rows and two columns that reports the number of false positives,
        false negatives, true positives, and true negatives.
    :return precision, recall, accuracy, f1
    """
    try:
        precision = confusion_matrix['TP'] / (confusion_matrix['TP'] + confusion_matrix['FP'])
        recall = confusion_matrix['TP'] / (confusion_matrix['TP'] + confusion_matrix['FN'])
        accuracy = (confusion_matrix['TP'] + confusion_matrix['TN']) / (confusion_matrix['TP'] + confusion_matrix['TN'] +
                                                                        confusion_matrix['FP'] + confusion_matrix['FN'])
        f1 = 2 * (recall * precision) / (recall + precision)

        return precision, recall, accuracy, f1
    except Exception:
        print('some element of the confusion matrix is equal to zero so the function cannot be applied.')


def check_causal_direction(relations_founded, edges, graph, anm_method=False):
    """
    :param relations_founded: tuple list
    :param edges: tuple list
    :param graph: networkx DiGraph
    :param anm_method: bool
    :return:
    """
    if anm_method:
        results = []
        for relation in relations_founded:
            # quando gli passo le dipendenze causali sono delle tuple
            # nel caso in cui ci siano più genitori tolgo l'elemento dalle coppie frequenti così
            # non inficia nel calcolo delle performance
            if re.match(r'\d*\w*,', relation[0]) is None:
                results.append(relation)
    else:
        results = relations_founded

    non_edges = list(elem for elem in nx.non_edges(graph))
    tn = list(set(non_edges) - set(results))

    possibilities = {'TP': 0, 'FP': 0,
                     'FN': 0, 'TN': len(tn)}
    no_match = {}
    for item in results:
        edge_found = False
        for edge in edges:
            if edge not in no_match:
                no_match[edge] = 0
            if edge == item:
                possibilities['TP'] += 1
                edge_found = True
            else:
                no_match[edge] += 1

        if not edge_found:
            possibilities['FP'] += 1

    for itemset, number in no_match.items():
        if number == len(results):
            possibilities['FN'] += 1

    return possibilities


class AutoVivification(dict):
    """
    Implementation of perl's autovivification feature.
    Generate a dict of dict.
    """

    def __getitem__(self, item):
        try:
            return dict.__getitem__(self, item)
        except KeyError:
            value = self[item] = type(self)()
            return value


def detect_cycles(graph):
    try:
        cycle = nx.find_cycle(graph)
    except nx.NetworkXNoCycle:
        return False, None
    return True, cycle


def delete_cycle(causal_relations_found, graph):

    has_cycle, cycle = detect_cycles(graph)
    while has_cycle:
        cycle_filter = dict()
        for nodes in cycle:
            # check self-loops
            if nodes[0] == nodes[1]:
                del causal_relations_found[nodes]
                graph = nx.DiGraph(list(causal_relations_found.keys()))
                has_cycle, cycle = detect_cycles(graph)
                continue
            # save only probabilities of edges in the cycle
            if nodes not in cycle_filter:
                cycle_filter[nodes] = 0.0
            cycle_filter[nodes] = causal_relations_found[nodes]

        node_to_delete = min(cycle_filter, key=cycle_filter.get)
        del causal_relations_found[node_to_delete]

        graph = nx.DiGraph(list(causal_relations_found.keys()))
        has_cycle, cycle = detect_cycles(graph)

    result = [tuple(elem) for elem in set(causal_relations_found)]
    return result


def createFolder(path):
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except OSError:
        print('Error: Creating directory. ' + path)


def randomDag(path, n_nodes=5, n_edges=2):
    """
    Create a random directed acyclic graph.
    :param path: string
        File path
    :param n_nodes: int, default = 5
        Number of nodes of the dag
    :param n_edges: int, default = 2
        Maximum number of nodes of the dag
    :return: networkx.DiGraph
    """

    graph = nx.gnm_random_graph(n_nodes, n_edges, directed=True)
    while not nx.is_directed_acyclic_graph(graph):
        graph = nx.gnm_random_graph(n_nodes, n_edges, directed=True)

    data = json_graph.node_link_data(graph)

    with open(os.path.join(path, 'ground_truth.json'), 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    return graph

