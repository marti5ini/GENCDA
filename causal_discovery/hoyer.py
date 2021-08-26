import os
import numpy as np
from operator import itemgetter
import matplotlib.pyplot as plt

import pandas as pd
from numpy.random import uniform
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

from causal_discovery.hsic import *
from utils import show_dependency, plot_predictionAndConfidenceInterval, all_matrices, is_acyclic, has_parents


class NonlinearANM:
    """
    Implementation of algorithm proposed on
    P. O. Hoyer, D. Janzing, J. M. Mooij, J. R. Peters, and B. Schölkopf.
    Nonlinear causal discovery with additive noise models. NIPS, 2009
    """

    def __init__(self, df):
        """
        :param df: numpy ndarray
        """
        self.df = df

    def _binary_test_causal_dependence(self, x, y, kernel, show_plots=False):
        """
        :param x: numpy ndarray (num_sample) x 1
        :param y: numpy ndarray (num_sample) x 1
        :param kernel: kernel instance
        :param show_plots: boolean, default = False
        :return: float
            p-value of independence test
        """
        # Instantiate a Gaussian Process model and fit to data using Maximum Likelihood Estimation of the parameters
        model = GaussianProcessRegressor(kernel=kernel).fit(x, y)
        residuals = y - model.predict(x)
        if show_plots:
            # Plot the function, the prediction and the 95% confidence interval based on the MSE
            plot_predictionAndConfidenceInterval(x, y, model)
            # Plot residuals
            plt.scatter(x, residuals)
            plt.show()
        # Statistical test of independence between residuals and independent variable
        return hsic_gam(residuals, x)

    def fit_bivariate(self, x, y, kernel=None, show_plots=False):
        """
        :param x: numpy array (num_sample) x 1
            column of the dataset
        :param y: numpy array (num_sample) x 1
            column of the dataset
        :param kernel: kernel instance
            the kernel specifying the covariance function of the GP.
            If None is passed, the kernel “1.0 * RBF(1.0)” is used as default.
        :param show_plots: boolean, default = False
            To show plots about prediction, confidence interval and residuals
        :return: list
            p-values between [two variables, first variable and its residuals, second variable and its residuals]
        """
        independence = hsic_gam(x, y)
        x_y = self._binary_test_causal_dependence(x, y, kernel, show_plots)
        y_x = self._binary_test_causal_dependence(y, x, kernel, show_plots)
        results = [independence, x_y, y_x]

        return results

    def _check_mutually_independence(self, X, res, alpha):
        """
        :param X: numpy ndarray (num_samples) x (num_parents)
        :param res: numpy array num_samples
        :return: tuple (boolean, float)
        """
        res = res.reshape(-1, 1)
        min_p = float('inf')

        for k in range(0, X.shape[1]):
            p_value = hsic_gam(X[:, [k]], res)
            if p_value < min_p:
                min_p = p_value
            if p_value < alpha:
                return False, p_value
        return True, min_p

    def _test_causal_dependence(self, parents, dependent, alpha, percentage_split, mutual=False):
        """
        :param parents: numpy ndarray (num_samples) x (num_parents)
            parents df
        :param dependent: numpy array num_samples
            dependent column
        :param alpha: float
            test threshold
        :param percentage_split: float in (0, 1)
        :param mutual: boolean, default = False
            Check if variables are mutually independent
        :return: tuple (boolean, float)
            If the model is accepted or not and the p-value

        """
        n_train_records = round(len(parents) * percentage_split)
        x_train, x_test = np.split(parents, [n_train_records])
        y_train, y_test = np.split(dependent, [n_train_records])

        for col in range(parents.shape[1]):
            if hsic_gam(parents[:, col].reshape(-1, 1), dependent.reshape(-1, 1)) > alpha:
                # The null hypothesis is rejected
                # Parents and dependent column are independent
                return False, -1

        model = GaussianProcessRegressor().fit(x_train, y_train)
        model.fit(x_train, y_train)
        residuals = y_test - model.predict(x_test)

        if mutual:
            mutually_independence, p_value = self._check_mutually_independence(x_test, residuals, alpha)
            return mutually_independence, p_value

        # Statistical test of independence between residuals and independent variable
        p_value = hsic_gam(x_test, residuals.reshape(-1, 1))

        if p_value > alpha:
            # The null hypothesis is accepted since the variables are independent
            return True, p_value
        # The null hypothesis is rejected as the variables are dependent
        return False, p_value

    def fit_multivariate(self, alpha, percentage_split, dag_sorting_type=np.min, mutual=False):
        """
        :param alpha: float
            test threshold
        :param percentage_split: float in (0, 1)
        :param dag_sorting_type: numpy function
            Type of aggregation of all the p-values of each causal dependency in the adjacency matrix.
            The result corresponds to the probability that the adjacency matrix matches the ground truth.
            It should be np.min() or np.max() or np.mean().
        :param mutual: boolean, default = False
            If check mutually independence or not.
        :return: list of tuples (ndarray, float)
            adjacency matrix, the probability that it corresponds to the ground-truth graph
        """
        n = self.df.shape[1]
        matrices = all_matrices(n)
        result = []
        tested_dependencies = dict()
        for matrix in matrices:
            p_values_matrix = np.zeros((n, n))
            if is_acyclic(matrix):
                check_model = True
                parents_list = has_parents(matrix)
                for vertex, parents in enumerate(parents_list):
                    if len(parents) > 0:
                        dependent_col = self.df[:, vertex]
                        parents_df = self.df[:, parents]
                        dependence = (str(parents_list), vertex)
                        # avoid repeating the statistical test of independence storing causal dependencies founded
                        # example: graph1 : w --> x,y e x,y --> z and graph2: x,y --> z
                        # x,y --> z tested just once
                        if dependence not in tested_dependencies:
                            causal_dependence = self._test_causal_dependence(parents_df, dependent_col, alpha,
                                                                             percentage_split, mutual)
                            tested_dependencies[dependence] = causal_dependence
                        else:
                            causal_dependence = tested_dependencies[dependence]
                        for elem in parents:
                            p_values_matrix[elem][vertex] = causal_dependence[1]
                        if not causal_dependence[0]:
                            check_model = False
                            break
                if check_model:
                    # To determine the simplest DAG that is consistent with the data,
                    # we store the minimum p-value among all those in the adjacency matrix.
                    # The probability that a DAG corresponds to the true causal structure
                    # is the same as the least probable causal dependency.
                    result.append((matrix, dag_sorting_type(p_values_matrix[np.nonzero(p_values_matrix)])))
        return result


if __name__ == '__main__':

    user_input = input('Would you like to try: bivariate or multivariate?')
    option1 = 'bivariate'
    option2 = 'multivariate'

    if user_input == option1:
        # Experiment on abalone dataset (https://archive.ics.uci.edu/ml/datasets/abalone)
        # Column: Rings, Length
        # Number of sample: 500
        directory = os.path.realpath(os.path.dirname(__file__))
        file_path = os.path.join(os.path.dirname(directory), "datasets", "abalone.csv")
        dataset = pd.read_csv(file_path)
        df = dataset.sample(500)
        first_variable, second_variable = df.iloc[:, [0]], df.iloc[:, [1]]

        kernel = 1.0 * RBF(length_scale=100.0, length_scale_bounds=(1e-2, 1e3)) + WhiteKernel(noise_level=1,
                                                                                              noise_level_bounds=(
                                                                                                  1e-10, 1e+1))
        df = NonlinearANM(df.values)
        result = df.fit_bivariate(first_variable.values, second_variable.values, kernel=kernel)
        print(result)

    elif user_input == option2:
        # Experiment on synthetic data used in the paper
        np.random.seed(1)
        n = 1000
        nx = uniform(-1, 1, n)
        ny = uniform(-1, 1, n)
        nz = uniform(-1, 1, n)
        w = uniform(-3, 3, n)
        x = w ** 2 + nx
        y = 4 * np.sqrt(np.absolute(w)) + ny
        z = 2 * np.sin(x) + 2 * np.sin(y) + nz

        matrix = np.column_stack((w, x, y, z))
        values = NonlinearANM(matrix)

        possible_dags = values.fit_multivariate(0.02, 0.7, np.max)
        ordered_result = sorted(possible_dags, key=itemgetter(1), reverse=True)

        res = []
        for adj_matrix, p_value in possible_dags:
            res.append(adj_matrix)
        columns = ['w', 'x', 'y', 'z']
        show_dependency(res, columns)
    else:
        raise ValueError('Digit a correct choice: bivariate or multivariate')
