import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from DBIndex import DBIndex
from sklearn.cluster import KMeans as KMEANS
from utils import visualize, normalize_columns
from sklearn.metrics import davies_bouldin_score
import logging
from itertools import combinations
from sklearn.decomposition import pca
from scipy.spatial.distance import cdist as euclidian_dist
from DBIndex import DBIndex
import fcmeans as fcm_lib

# monkey path round to identity function
# done due to laziness
_round = lambda x, y: x


class FCM():
    def __init__(self, n_clusters, m=2,  n_iter=100, random_state=120):
        self.k = n_clusters
        self.m = m
        self.n_iter = n_iter
        self.seed = random_state
        logging.basicConfig(level=logging.INFO)
        self.log = logging

    def init_centroids(self, data, k, seed):
        return data.sample(k, random_state=seed).values

    def init_membership_random(self, data, centroids, seed):
        """
        :param num_of_points:
        :return: membership matrix
        """
        np.random.seed(seed)
        n_points = len(data)
        n_clusters = len(centroids)
        membership_matrix = {centroid: np.zeros(n_points) for centroid in centroids}
        for _idx, row in enumerate(data.values):
            row_total = 0
            # for centroid in centroids:
            random_value = np.random.randint(0, n_clusters)
            row_total += 1
            membership_matrix[centroids[random_value]][_idx] = 1 if row_total <= 1 else 0
        return pd.DataFrame(membership_matrix)

    def compute_centroids(self, data, old_membership_matrix):
        w_k_sqaured = np.power(old_membership_matrix.values, self.m)
        # print(w_k_sqaured.shape)
        # print(np.matmul(data.values.T, w_k_sqaured))
        return np.transpose(np.matmul(data.values.T, w_k_sqaured) / np.sum(w_k_sqaured, axis=0))

    def update_memberships(self, dist_to_centroid_df, centroids):
        updated_memberships = pd.DataFrame(columns=centroids)
        for centroid in centroids:
            ratio_sum = np.sum(
                [(dist_to_centroid_df[centroid] / dist_to_centroid_df[centroid_i])  for centroid_i in centroids],
                axis=0)
            updated_memberships[centroid] = np.nan_to_num((1 / ratio_sum))
        return updated_memberships

    def fit(self, data):
        centroids = self.init_centroids(data, self.k, self.seed)
        #self.log.info(centroids)
        centroids = [tuple(centroid) for centroid in centroids]
        # _data = np.array([[0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0,
        #                   0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1,
        #                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1,
        #                   1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1],
        #                  [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0,
        #                   0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0,
        #                   1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0,
        #                   0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
        #                  [1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
        #                   1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0,
        #                   0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0,
        #                   0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0]])
        # membership_matrix = pd.DataFrame(_data.T, columns=centroids)
        membership_matrix = self.init_membership_random(data, centroids, self.seed)
        #print(membership_matrix.shape)
        centroids_old = centroids
        _iter = 0
        while (_iter < self.n_iter):
            # old_memberships = membership_matrix.copy()

            centroids_new = self.compute_centroids(data, membership_matrix)
            centroids_new = [tuple(centroid) for centroid in centroids_new]
            # centroids_new = centroids
            dist_matrix = euclidian_dist(data, centroids_new)
            dist_matrix = dist_matrix ** 2
            dist_to_centroid_df = pd.DataFrame(dist_matrix, columns=centroids_new)

            updated_memberships = self.update_memberships(dist_to_centroid_df, centroids_new)
            #print(u)
            membership_matrix = updated_memberships
            error = 1e-6
            centroids_new = self.compute_centroids(data, membership_matrix)
            #print(_iter)
            if(np.alltrue(np.isclose(centroids_new, centroids_old, rtol=error ,atol=error))):
                break;
            centroids_old = centroids_new
            _iter +=1
        print("stopped at ", _iter-1, " for K = ", self.k )
        self.log.info(list(membership_matrix))
        return membership_matrix

    def hard_cluster_memberships(self, data, membership_matrix):
        # returns keys of dict which are centroids
        final_centroids = list(membership_matrix)
        labels = pd.DataFrame()
        labels["cluster_id"] = np.zeros(shape=(data.shape[0]), dtype=np.int64)
        centroid_memberships = {centroid: [] for centroid in final_centroids}
        for _idx, data in enumerate(data.values):
            centroid_idx = np.argmax(membership_matrix.values[_idx])
            centroid_memberships[final_centroids[centroid_idx]].append(data)
            labels["cluster_id"].loc[_idx] = centroid_idx
        return centroid_memberships, labels


if __name__ == "__main__":
    #load data
    bsom_data = pd.read_csv("BSOM_DataSet_revised.csv")
    requried_cols = ["all_NBME_avg_n4", "all_PIs_avg_n131", "HD_final"]
    bsom_clean = normalize_columns(bsom_data[requried_cols])


    fcm = FCM(n_clusters=3, n_iter=100, random_state=120)
    membership_matrix = fcm.fit(bsom_clean)
    # for using centroid as key in dict, list can;t be used as key so we use tuple
    # final_centroids = [tuple(centroid) for centroid in final_centroids]
    # centroid_memberships = {centroid:[] for centroid in final_centroids}
    # for _idx, data in enumerate(bsom_clean.values):
    #     centroid_idx = np.argmax(membership_matrix.values[_idx])
    #     centroid_memberships[final_centroids[centroid_idx]].append(data)

    centroid_memberships,labels = fcm.hard_cluster_memberships(bsom_clean, membership_matrix)
    print(centroid_memberships)

    #membership_matrix =
    #print(fcm.fit(bsom_clean)[0])
    print("********* k")
    dbi_score = DBIndex(centroid_memberships)
    print(dbi_score.compute__DBIndex())

    """
    Question 3a
    k=2
    columns considered = ["all_NBME_avg_n4", "all_PIs_avg_n131", "HD_final", "all_irats_avg_n34", "HA_final"]
    """
    bsom_data = pd.read_csv("BSOM_DataSet_revised.csv")
    requried_cols = ["all_NBME_avg_n4", "all_PIs_avg_n131", "HD_final", "all_irats_avg_n34", "HA_final"]
    bsom_new_feature = normalize_columns(bsom_data[requried_cols])

    fcm = FCM(n_clusters=2, random_state=120)
    centroid_memberships = fcm.fit(bsom_new_feature)
    centroid_memberships, labels = fcm.hard_cluster_memberships(bsom_new_feature, centroid_memberships)
    #[(0.7671764705882353, 0.7299532922681461, 0.7692810457516339, 0.6582167832167831, 0.7494553376906319),
    # (0.4369361702127658, 0.39192475806588767, 0.4884160756501182, 0.34650864683597904, 0.5350669818754924)]
    print(list(centroid_memberships))
    dbi_score = DBIndex(centroid_memberships)
    print(dbi_score.compute__DBIndex())


    # plotting Davies Bouldin indices for every configuration of K
    fcms_arr = []
    #fcm_preds= []
    DB_Indices = []
    DB_Scores=[]
    for k in range(2,11):
        fcms_arr.append(FCM(n_clusters=k, random_state=120))
    fcm_preds = [fcms.hard_cluster_memberships(bsom_new_feature,fcms.fit(bsom_new_feature)) for fcms in fcms_arr]

    i=2
    for centroid_membership, labels in fcm_preds:
        dbi = DBIndex(centroid_membership)
        score = dbi.compute__DBIndex()
        print(i, score)
        DB_Indices.append(score)
        DB_Scores.append(davies_bouldin_score(bsom_new_feature, np.array(labels["cluster_id"])))
        i +=1

    plt.plot(range(2, 11), DB_Indices, label="DB_indices")
    plt.plot(range(2, 11), DB_Scores, label="DB indices using lib")
    plt.xlabel("n_clusters")
    plt.ylabel("DB Index")
    plt.title("n_clusters vs DB index")
    plt.legend()
    plt.show()


    # benchmarking
    for k in range(2,11):
        benchmark = fcm_lib.FCM(n_clusters=k, random_state=120, max_iter=100)
        print(k, benchmark.fit(bsom_new_feature).centers)
