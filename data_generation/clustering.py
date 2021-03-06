import numpy as np
from kneed import KneeLocator
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def kmeans(causal_df, kmax=10):
    """
    K-Means clustering.

    :param causal_df: pandas Dataframe
        Dataset generated from a DAG
    :param kmax: integer, default = 10
        The maximum number of clusters to test
    :return: tuple (k-means fitted with optimal k, sse)
    """
    # Standardization before clustering
    scaled_data = StandardScaler().fit_transform(causal_df.values)
    sse = []
    kmeans_class = {}
    for k in range(2, kmax + 1):
        if k not in kmeans_class:
            kmeans_class[k] = None
        # Compute clustering
        kmeans = KMeans(n_clusters=k).fit(scaled_data)
        # Sum of squared distances of samples to their closest cluster center
        sse.append(kmeans.inertia_)
        # Save fitting results and sse
        kmeans_class[k] = (kmeans, kmeans.inertia_)

    # Determining the elbow point in the SSE curve
    n_of_cluster = KneeLocator(range(2, kmax + 1), sse, curve="convex", direction="decreasing").elbow

    # Case without elbow point
    if n_of_cluster is None:
        n_of_cluster = kmax

    return kmeans_class[n_of_cluster]


def calculate_see(kmeans, generate_df):
    """
    Compute sum of squared distances of samples from random distributions
    to their closest cluster center.

    :param kmeans: numpy ndarray
        Fitted estimator with optimal number of clusters
    :param generate_df: pandas DataFrame
        Dataframe generated by random distributions
    :return: sse, float
    """
    # Standardize data before clustering
    scaled_generate = StandardScaler().fit_transform(generate_df.values)
    # Get centroids
    centroids = kmeans.cluster_centers_
    # Predict of random data
    pred_clusters = kmeans.predict(scaled_generate)
    # Sum of squared distances of samples to their closest cluster center
    curr_sse = 0
    for i, cluster in enumerate(pred_clusters):
        curr_center = centroids[cluster]
        curr_sse += np.sum((scaled_generate[i] - curr_center) ** 2)

    return curr_sse
