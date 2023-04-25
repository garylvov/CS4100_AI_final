import numpy as np
from synthetic_data_gen import SparsePointAssocDataset
from sklearn.cluster import KMeans, DBSCAN

'''
Authored by Gary Lvov
'''
n_clusters = 10
errors = []

spad = SparsePointAssocDataset(clustering=True, num_samples=1000)

obs_names = set()
t_names = set()

for idx in range(spad.num_geom):
    obs_names.add('obs_' + str(idx))

for idx in range(spad.num_geom):
    t_names.add('t_' + str(idx))

# K-Means Clustering
errors = []
for n_c in range(1,n_clusters):
    error = 0
    for dict in spad.point_dicts:
        points = []
        ts = []

        for key, value in dict.items():
            if key in obs_names:
                points.append(value)
            if key in t_names:
                ts.append(value)

        points = np.array(points)
        points = points.reshape(points.shape[0] * points.shape[1], 3)
        np.random.shuffle(points)

        kmeans = KMeans(n_clusters = n_c, random_state=0, n_init="auto").fit(points)
        total_t_kmeans = sum(kmeans.cluster_centers_)
        total_t_groundtruth = sum(ts)

        error += abs(total_t_kmeans - total_t_groundtruth)
    errors.append(error)

x_error = [x[0]/spad.num_geom for x in errors]
y_error = [x[1]/spad.num_geom for x in errors]
z_error = [x[2]/spad.num_geom for x in errors]

abs_error = [sum(error) for error in errors]
avg_error = [error/spad.num_geom for error in abs_error]

best_cluster_idx = avg_error.index(min(avg_error))

print(f'Optimal number of K-means clusters for {spad.num_geom}'  +
      f' geometries is {best_cluster_idx + 1}')
print(f'Average error {avg_error[best_cluster_idx]}')
print(f'avg x error: {x_error[best_cluster_idx]} avg y error: \
      {y_error[best_cluster_idx]} avg z error: {z_error[best_cluster_idx]}')

# DBSCAN Clustering
lower_bound = .01
upper_bound = 5
granularity = 10
eps_values = np.linspace(lower_bound, upper_bound, granularity)

errors = []
for eps_idx in range(granularity):
    eps = eps_values[eps_idx]
    for min_sample in range(1, spad.num_geom):
        error = 0
        for dict in spad.point_dicts:
            points = []
            ts = []

            for key, value in dict.items():
                if key in obs_names:
                    points.append(value)
                if key in t_names:
                    ts.append(value)

            points = np.array(points)
            points = points.reshape(points.shape[0] * points.shape[1], 3)
            dbscan = DBSCAN(eps=eps, min_samples=min_sample).fit(points)
            labels = dbscan.labels_
            
            # calculate the "center" of each cluster
            centers = []
            for label in set(labels):
                if label != -1:
                    center = np.mean(points[labels == label], axis=0)
                    centers.append(center)
            
            total_t_dbscan = sum(centers)
            total_t_groundtruth = sum(ts)

            error += abs(total_t_dbscan - total_t_groundtruth)

    errors.append(error)

x_error = [x[0]/spad.num_geom for x in errors]
y_error = [x[1]/spad.num_geom for x in errors]
z_error = [x[2]/spad.num_geom for x in errors]

print(errors)