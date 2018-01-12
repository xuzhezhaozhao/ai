import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN

data = np.loadtxt('records_tags.in.shuf.vec', dtype=np.str, delimiter=' ')

data = data[:, :-1].astype(np.float)

y = KMeans(n_clusters=300).fit_predict(data)
# y = DBSCAN(eps=0.1, metric='euclidean', n_jobs=-1).fit_predict(data)
d = np.loadtxt('records_tags.in.shuf.dict', dtype=np.str)
num = y.shape[0]
clusters = dict()
for i in range(num):
    label = y[i]
    if label not in clusters:
        clusters[label] = list()
    clusters[label].append(d[i])

for i in clusters[0]:
    print i
