import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity


def fit_kde(data):
    """
    Compute a gaussian kernel density estimate with the best bandwidth.
    :param data: column of a pandas DataFrame
    :return: estimator that was chosen by the search
    """
    bandwidth = [0.01, 0.1, 0.2, 0.5, 1, 1.5, 2]
    kde = KernelDensity(kernel='gaussian')
    grid = GridSearchCV(kde, {'bandwidth': bandwidth})
    grid.fit(data)
    return grid.best_estimator_


def get_statistics(dataframe, dataframe1, linspace_samples=100):
    """
    Compute pairwise sum of squared errors of columns
    using the Kernel Density Estimation (KDE).

    :param dataframe: pandas DataFrame
    :param dataframe1: pandas DataFrame
    :param linspace_samples: integer
        Number of sample of a linespace used to calculate the Probability Density Function
    :return: dict
    """
    result = {}
    sse = []
    rmse = []

    for index in range(len(dataframe.columns)):
        col_df = dataframe.iloc[:, index]
        col_df1 = dataframe1.iloc[:, index]
        data1 = col_df.values.reshape(-1, 1)
        data2 = col_df1.values.reshape(-1, 1)

        # Fit Kernel Density Function
        kde1 = fit_kde(data1)
        kde2 = fit_kde(data2)
        x_grid = np.linspace(min(np.min(data1), np.min(data2)),
                             max(np.max(data1), np.max(data2)), linspace_samples).reshape(-1, 1)

        # Probability Density Function
        pdf1 = np.exp(kde1.score_samples(x_grid))
        pdf2 = np.exp(kde2.score_samples(x_grid))

        # RMSE
        rmse.append(mean_squared_error(pdf1, pdf2, squared=False))
        # SSE
        sse.append(np.sum(np.power(pdf1 - pdf2, 2.0)))

    result['rmse'] = {'mean': np.mean(rmse)}
    result['sse'] = {'mean': np.mean(sse)}

    return result


def plotKDE(column1, column2, label1='Original Data', label2='GENCDA'):
    """
    :param column1:
        Single column of a pandas.Dataframe
        representing the original dataset
    :param column2:
        Single column of a pandas.Dataframe
        representing the synthetic dataset generated.
    :param label1: string, default = 'Original Data'
        Name of dataset
    :param label2: string, default = 'GENCDA'
        Name of synthetic data generator used to produce the synthetic dataset.
    :return:
    """

    data1 = column1.values.reshape(-1, 1)
    data2 = column2.values.reshape(-1, 1)

    # Fit Kernel Density Function
    kde1 = fit_kde(data1)
    kde2 = fit_kde(data2)
    x_grid = np.linspace(min(np.min(data1), np.min(data2)),
                         max(np.max(data1), np.max(data2)), 1000).reshape(-1, 1)

    # Probability Density Function
    pdf1 = np.exp(kde1.score_samples(x_grid))
    pdf2 = np.exp(kde2.score_samples(x_grid))

    # Plot pdf
    plt.plot(x_grid[:, 0], pdf1, linewidth=2, label=label1)
    plt.plot(x_grid[:, 0], pdf2, linewidth=2, label=label2)

    plt.title(f'RMSE: {mean_squared_error(pdf1, pdf2, squared=False)},'
                 f' SSE: {np.sum(np.power(pdf1 - pdf2, 2.0))}')
    plt.legend(fontsize=13)
    plt.show()

    return