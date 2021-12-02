import os
import numpy as np
from operator import itemgetter
import matplotlib.pyplot as plt

import pandas as pd
from numpy.random import uniform
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

from causal_discovery.hsic import *
from utils import show_dependency, all_matrices, is_acyclic, has_parents


class NonlinearANM:
    """
    Implementation of algorithm proposed on
    P. O. Hoyer, D. Janzing, J. M. Mooij, J. R. Peters, and B. Schölkopf.
    Nonlinear causal discovery with additive noise models. NIPS, 2009
    https://papers.nips.cc/paper/3548-nonlinear-causal-discovery-with-additive-noise-models.pdf
    """

    def __init__(self, training_data):
        """
        :param training_data: pandas.DataFrame
        """
        self.data = training_data
        self.features_names = list(self.data.columns)
        self.index_col1 = self.index_col2 = self.x = self.y = self.model_fm = \
            self.model_bm = self.residuals_fm = self.residuals_bm = None

    def _binary_test_causal_dependence(self, kernel, direction):
        """
        :param x: numpy ndarray (num_sample) x 1
        :param y: numpy ndarray (num_sample) x 1
        :param kernel: kernel instance
        :param show_plots: boolean, default = False
        :return: float
            p-value of independence test
        """
        if direction == 'forward':
            var1, var2 = self.x, self.y
        else:
            var1, var2 = self.y, self.x
        # Instantiate a Gaussian Process model and
        # fit to data using Maximum Likelihood Estimation of the parameters
        model = GaussianProcessRegressor(kernel=kernel).fit(var1, var2)
        # Calculate the the corresponding residual n = y - f(x)
        residuals = var2 - model.predict(var1)
        # Statistical test of independence between residuals and independent variable
        return model, residuals, hsic_gam(residuals, var1)

    def fit_bivariate(self, idx_var1, idx_var2, kernel=None):
        """
        :param idx_var1: int, default = 0
            Index of the first variable
        :param idx_var2: int, default = 1
            Index of the second variable
        :param kernel: kernel instance
            the kernel specifying the covariance function of the GP.
            If None is passed, the kernel “1.0 * RBF(1.0)” is used as default.
        :param show_plots: boolean, default = False
            To show plots about prediction, confidence interval and residuals
        :return: list
            p-values between [two variables, first variable and its residuals, second variable and its residuals]
        """
        self.index_col1, self.index_col2 = idx_var1, idx_var2
        self.x, self.y = self.data.iloc[:, self.index_col1].values.reshape(-1, 1), self.data.iloc[:, self.index_col2].values.reshape(-1, 1)
        p_value_general = hsic_gam(self.x, self.y)
        # Check Forward model (FM)
        self.model_fm, self.residuals_fm, p_value_fm = self._binary_test_causal_dependence(kernel, direction='forward')
        # Check Backward model (BM)
        self.model_bm, self.residuals_bm, p_value_bm = self._binary_test_causal_dependence(kernel, direction='backward')

        return p_value_general, p_value_fm, p_value_bm

    def evaluate(self, p_value_general, p_value_fm, p_value_bm, alpha=0.05):
        """"
        :param p_value_general: float,
            Result of independence test between two variables
        :param p_value_fm: float,
            Result of independence test between the first variable and its residuals
        :param p_value_bm: float,
            Result of independence test between the second variable and its residuals
        :param alpha: float, default = 0.05
            Test threshold
        :return: string
        """
        # Case 1
        if p_value_general > alpha:
            print(f'{self.features_names[self.index_col1]} and {self.features_names[self.index_col2]} '
                  f'are statistically independent')
        # Case 2
        elif p_value_fm < alpha < p_value_bm:
            print(f'{self.features_names[self.index_col1]} causes {self.features_names[self.index_col2]}')
        # Case 3
        elif p_value_fm > alpha > p_value_bm:
            print(f'{self.features_names[self.index_col1]} causes {self.features_names[self.index_col2]}')
        # Case 4
        elif p_value_fm > alpha < p_value_bm:
            print('Both directional models are accepted. '
                  'We conclude that either model may be correct but we cannot infer it from the data.')
        # Case 5
        else:
            print('Neither direction is consistent with the data '
                  'so the generating mechanism cannot be described using this model.')
        return

    @staticmethod
    def plot(variable1, variable2, name_var1, name_var2, model, residuals):

        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Plot the function, the prediction and the 95% confidence interval based on the MSE
        x_pred = np.linspace(min(variable1), max(variable1), 1000).reshape(-1, 1)
        y_pred, sigma = model.predict(x_pred, return_std=True)
        sigma = sigma.reshape(-1, 1)

        ax1.plot(variable1, variable2, 'r.', markersize=5, label='Observations')
        ax1.plot(x_pred, y_pred, 'b-', label='Prediction')
        ax1.fill_between(x_pred.reshape(-1),
                         np.reshape(y_pred - 1.9600 * sigma, -1),
                         np.reshape(y_pred + 1.9600 * sigma, -1),
                         alpha=.5, label='95% confidence interval')
        ax1.set_xlabel(name_var1)
        ax1.set_ylabel(name_var2)

        # Plot residuals
        ax2.scatter(variable1, residuals)
        ax2.set_xlabel(name_var1)
        ax2.set_ylabel('Residuals of' + name_var1)
        return

    def show_plots(self):
        name_x = self.features_names[self.index_col1]
        name_y = self.features_names[self.index_col2]
        print('Forward Model ' + '(' + name_x + ' , ' + name_y + ')')
        self.plot(self.x, self.y, name_x, name_y, self.model_fm, self.residuals_fm)
        plt.show()
        print('Backward Model ' + '(' + name_y + ' , ' + name_x + ')')
        self.plot(self.y, self.x, name_y, name_x, self.model_bm, self.residuals_bm)

    @staticmethod
    def _check_mutually_independence(X, res, alpha):
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
        n = len(self.features_names)
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
                        dependent_col = self.data.iloc[:, vertex].values
                        parents_df = self.data.iloc[:, parents].values
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
