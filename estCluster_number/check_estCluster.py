# -*- coding: utf-8 -*-

"""
FileName:               check_estCluster.py
Author Name:            Arun M Saranathan
Description:            This code file is used to create a simulated dataset made up of gaussian clusters to check the
                        performance of algorithm proposed in [1], to estimate the number of clusters in the data

                        [1] Han, K., Vedaldi, A., and Zisserman, A., 2019, "Learning to discover novel visual
                        categories via deep transfer clustering." In Proceedings of the IEEE/CVF International
                        Conference on Computer Vision, (pp. 8401-8409).

Date Created:           10th June 2021
Last Modified:          14th June 2021
"""

import numpy as np
import math
from sklearn.datasets import make_blobs

from estCluster_number import estCluster_number
"""from sklearn.metrics import adjusted_mutual_info_score
from matplotlib import pyplot as plt

from SemiSupervisedKMeans.semiSupervised_kmeans import KMeans"""

if __name__ == "__main__":
    'Create a dataset with n clusters'
    nClust = 10
    X, y_true = make_blobs(n_samples=10000, centers=nClust, cluster_std=0.50, random_state=0)
    'From this dataset throw away information on some of the clusters'
    y_true_red = y_true.copy()
    y_true_red[y_true_red >= 7] = -1

    '------------------------------------------------------------------------------------------------------------------'
    'GET SOME KNOWN DATA'
    known_data = []
    known_classes = [0, 1, 3, 5]
    for ii in known_classes:
        'Find some example of chosen supervised class'
        temp = np.random.choice(np.where(y_true_red == ii)[0], 40)
        'Add them to the list of known data'
        known_data += [temp]

    'Create an object to estimate the number of clusters'
    obj1 = estCluster_number(k_min=4, k_max=16, step_size=1, metric_name='euclidean', verbose=False)
    kmeans_perf_simClust = obj1.est_numClusters(X, y_true_red, known_data=known_data)
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