# -*- coding: utf-8 -*-

"""
FileName:               estCluster_number.py
Author Name:            Arun M Saranathan
Description:            This code file is used to estimate the number of clusters in the data by using the technique
                        using the method described in [1].

                        [1] Han, K., Vedaldi, A., and Zisserman, A., 2019, "Learning to discover novel visual
                        categories via deep transfer clustering." In Proceedings of the IEEE/CVF International
                        Conference on Computer Vision, (pp. 8401-8409).

Date Created:           7th June 2021
Last Modified:          7th June 2021
"""

import numpy as np
import math
from fastdist import fastdist as fd
from sklearn.metrics import silhouette_score
from scipy.optimize import linear_sum_assignment as linear_assignment
from tqdm import tqdm

from openMax.data_utilities import  get_openSet
from hsiUtilities.crism_ganProc_utils import crism_ganProc_utils
from SemiSupervisedKMeans.semiSupervised_kmeans import KMeans

class estCluster_number(object):
    def __init__(self,k_min, k_max, step_size=2, metric_name='euclidean', verbose=False):
        """
        :param k_min: [int]
        Minimum number of clusters to be considered for the data.

        :param k_max: [int]
        Maximum number of clusters to be considered for the data.

        :param step_size [int] (Default: 5)
        The step size in which the number of clusters are varied for the K-means.

        :param metric_name: [string] (Default: 'euclidean')
        This string argument decides the distance metric used. Valid inputs are:
            (a) 'euclidean' (default)
            (b) 'minkowski'
            (c) 'seculidean'
            (d) 'chebyshev'
            (e) 'mahalanobis'
            (f) 'cosine'

        :param verbose: [Boolean] (Default: False)
        A flag used to decide if the processing results are displayed.
        """

        assert isinstance(k_min, int) and (k_min > 0), 'k_min must be a nonzero positive integer'
        assert isinstance(k_max, int) and (k_max >= k_min), 'k_max must be a nonzero integer >= k_min'
        assert isinstance(step_size, int), "Step size must be an integer"
        assert metric_name in ['euclidean', 'manhattan', 'minkowski',
                               'sqeuclidean', 'chebyshev', 'mahalanobis', 'cosine', 'cityblock'], \
            "Invalid distance metric."
        assert isinstance(verbose, bool), "The verbose flag can only be True or False."

        #'The list of possible distance functions is'
        #func_dict = {'euclidean': fd.euclidean, 'minkowski': fd.minkowski, 'sqeuclidean': fd.sqeuclidean,
        #             'chebyshev': fd.chebyshev, 'mahalanobis': fd.mahalanobis, 'cosine': fd.cosine,
        #             'cityblock': fd.cityblock}

        self.__k_min = k_min
        self.__k_max = k_max
        self.__step_size=step_size
        self.__metric_name = metric_name
        #self.metric = func_dict.get(metric_name)
        self.verbose = verbose

        'Check the metric is available'
        self._validate_metric()

    def _validate_metric(self):
        """
        Validates that the prescribed metric is functional.

        :return:
        """
        try:
            fd.matrix_to_matrix_distance(np.random.RandomState(seed=0).rand(10, 50),
                                         np.random.RandomState(seed=0).rand(100, 50))
        except Exception as e:
            print(e)

        return

    def est_numClusters(self, data, labels, known_data=None):
        """

        :param data: [ndarray: nSamples X nDims]
        The data on which the number of clusters are to be estimated.

        :param labels: [ndarray: nSamples]
        The samples for which labeled data is provided. The accuracy of these samples is going to be used for deciding
        the number of clusters. The samples for which class labels are not available the model

        :param known_data: [2D list] (Default: None)
        A 2D array of indicies which indicates the labels of the known classes. Classes with no labeled samples are
        represented by an empty list.

        :return:
        """

        assert len(data.shape) >= 2, "The data has to be a numpy matrix with atleast 2 dimensions, with the first " \
                                "dimension being the number of samples"
        assert (len(labels.shape) == 1) and (labels.shape[0] == data.shape[0]), "The labels must be a 1D matrix with a " \
                                                                           "single value for each sample"

        'The number of known classes is'
        self._knwnClasses = np.max(labels) + 1

        'Perform the semi-supervised k-means and get the labels'
        k_means_perf_params = []
        for kk in tqdm(range(self.__k_min, self.__k_max, self.__step_size)):
            """print('Processing with {} clusters.\n'.format(kk))"""
            self.kmeans_semi = KMeans(k=kk, known_data=known_data, verbose=self.verbose)
            kmeans_semi_results = self.kmeans_semi.fit_predict(data)

            'Now perform the perform the clustering accuracy for the additional known classes in the probe set'
            knwnClust_acc, unknwnClust_cvi = self._eval_clust_metrics(data, labels, kmeans_semi_results)

            k_means_perf_params.append([kk, knwnClust_acc, unknwnClust_cvi])

        return k_means_perf_params

    def _eval_clust_metrics(self, data, labels, clust_labels):
        """
        This function is used to check the accuracy of the clustering interms of the labeled samples (from both the
        probe set-both anchor and validation portions). The points are considered correctly clustered if a majority of
        the labeled points are in the same cluster

        :param data: [ndarray: nSamples X nDims]
        The data on which the number of clusters are to be estimated.

        :param labels: [ndarrau: nSamples]
        The label set for the classes which are labeled.

        :param clust_labels: [ndarray: nSamples]
        The is a 1D array with the estimated cluster labels

        :return: returns mean accuracy for clusters which contain the labeled classes
        """

        'Find the indicies of the labeled and unlabeled data'
        probe_set_ind = np.where(labels != -1)[0]
        nonprobe_set_ind = np.where(labels == -1)[0]

        'For the samples in the labeled probe set (both anchor and validation) get a measure of the accuracy'
        probe_acc = self.cluster_acc(labels[probe_set_ind], clust_labels[probe_set_ind])

        'For the unlabeled samples get a measure of the Cluster validaity by using metrics such as silhoette score'
        cvi_sil = silhouette_score(data[nonprobe_set_ind], clust_labels[nonprobe_set_ind])

        return probe_acc, cvi_sil


    def cluster_acc(self, y_true, y_pred):
        """
        This function is used to identify the cluster accuracy for the method to estimate the number of clusters in [1].
        This implementation is the same as the one the authors provide @:
        github.com/k-han/DTC/blob/ --> utils/utils.py --> cluster_acc

        :param y_true: [ndarrau: nSamples]
        The true labels for the samples.

        :param y_pred: [ndarray: nSamples]
        The is a 1D array with the estimated cluster labels

        :return: a measure of clustering accuracy
        """
        'Get the true and generated cluster labels'
        y_true = y_true.astype(np.int64)
        assert y_pred.size == y_true.size

        'Create a graph matrix to track cluster assignments'
        D = max(y_pred.max(), y_true.max()) + 1
        w = np.zeros((D, D), dtype=np.int64)
        'Get the class distribution for each cluster in the data'
        for ii in range(y_pred.size):
            w[y_pred[ii], y_true[ii]] += 1

        'Identify the best assignment to each cluster by solving the linear sum assignment problem'
        ind = linear_assignment(w.max() - w)
        sum = 0
        for ii in range(len(ind[0])):
            sum += w[ind[0][ii], ind[1][ii]]

        return (sum * 1.0 / y_true.size)


if __name__ == "__main__":
    'Get some sample from known and unknown classes'
    x, y = get_openSet(nSamples=20000)
    y = np.argmax(y, axis=1)

    'Save the test data so I can use in other tests'
    with open("trail.npy", "wb") as f:
        np.save(f, x)
        np.save(f, y)

    y_probe = y.copy()
    y_probe[y_probe >= 7] = -1

    known_classes = [0, 1, 2, 3]
    known_data = []
    for ii in known_classes:
        'Find some example of chosen supervised class'
        temp = np.where(y_probe == ii)[0]
        'Add them to the list of known data'
        known_data += [temp]

    'Create the feature extractor of interest'
    disRep = crism_ganProc_utils().create_rep_model()

    'Get the feature space representation for the data'
    x_rep = disRep.predict(x)
    x_rep = x_rep.astype(np.float64)

    'Create an object to estimate the number of clusters'
    obj1 = estCluster_number(k_min=4, k_max=20, step_size=1, metric_name='euclidean', verbose=False)

    kmeans_perf_simClust = obj1.est_numClusters(x_rep, y_probe, known_data=known_data)
    kmeans_perf_simClust = np.asarray(kmeans_perf_simClust)

    'Find the location of the best performance in terms of both classification performance'
    max_acc = np.where(kmeans_perf_simClust[:, 1] == max(kmeans_perf_simClust[:, 1]))[0]
    max_cvi = np.where(kmeans_perf_simClust[:, 2] == max(kmeans_perf_simClust[:, 2]))[0]

    k_acc = kmeans_perf_simClust[max_acc[0], 0]
    k_cvi = kmeans_perf_simClust[max_cvi[-1], 0]

    k_pred = 0.5 * (k_acc + k_cvi)
    ind_pred = np.where(kmeans_perf_simClust[:, 0] == math.ceil(k_pred))[0]

    print('Based on performance the number of clusters is {}'.format(math.ceil(k_pred)))
    print('Accuracy: {:.3f}\t CVI: {:.3f}\n'.format((kmeans_perf_simClust[ind_pred, 1]).min(),
                                                    (kmeans_perf_simClust[ind_pred, 2]).min()))







