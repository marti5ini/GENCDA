import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from utils import AutoVivification
import matplotlib
import matplotlib.pyplot as plt


def lof(dataframe1, dataframe2, n_neighbors=10):
    """
    Compute Local Outlier Factor, an unsupervised anomaly detection method.

    The number of neighbors considered, (parameter n_neighbors) is typically chosen:
    1) greater than the minimum number of objects a cluster has to contain, so that other objects can be local outliers
    relative to this cluster;
    2) smaller than the maximum number of close by objects that can potentially be local outliers.
    In practice, such information are generally not available, and taking
    n_neighbors=20 appears to work well in general.
    (https://scikit-learn.org/stable/auto_examples/neighbors/plot_lof_novelty_detection.html?highlight=lof)


    :param dataframe1: pandas DataFrame
    :param dataframe2: pandas DataFrame
    :param n_neighbors: integer, default = 10
        Number of neighbors
    :return: ndarray of shape (n_samples,)
        The shifted opposite of the Local Outlier Factor of each input samples.
        The lower, the more abnormal.
        Negative scores represent outliers, positive scores represent inliers.
    """
    result = AutoVivification()
    clf = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True)
    # Fit on causal dataframe
    clf.fit(dataframe1.values)
    # Predict on random dataframe
    y_pred = clf.predict(dataframe2.values)
    # The number of errors
    n_error_test = y_pred[y_pred == -1].size
    # Shifted opposite of the LOF of X. The shift offset allows a zero threshold for being an outlier.
    # The argument X is supposed to contain new data.
    # Also, the samples in X are not considered in the neighborhood of any point.
    decision = clf.decision_function(dataframe2.values)
    # Save result
    result['n_error_test'] = n_error_test
    result['stats'] = {'mean': np.mean(decision), 'max': np.amax(decision), 'min': np.amin(decision),
                       'std': np.std(decision), 'median': np.median(decision)}
    return result


def plotLOF(column1, column2, n_neighbors=10, contamination=0.1):
    clf = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True, contamination=contamination)
    clf.fit(column1.values)
    y_pred_test = clf.predict(column2.values)
    n_error_test = y_pred_test[y_pred_test == -1].size
    n_points = y_pred_test[y_pred_test == 1].size

    outlier_index = np.where(y_pred_test == -1)
    inlier_index = np.where(y_pred_test == 1)

    s = 40
    b1 = plt.scatter(column1.iloc[:, 0], column1.iloc[:, 1], c='white', s=s, edgecolors='k')

    outlier_values = column2.iloc[outlier_index].values
    inlier_values = column2.iloc[inlier_index].values

    b2 = plt.scatter(outlier_values[:, 0], outlier_values[:, 1], color="r")
    b3 = plt.scatter(inlier_values[:, 0], inlier_values[:, 1], color="b")

    plt.title(f'# outlier: {n_error_test}, #inlier: {n_points}')

    plt.legend([b1, b2, b3],
               ["data", "outlier", "inlier"],
               loc="upper left",
               prop=matplotlib.font_manager.FontProperties(size=11))

    plt.show()

    return
