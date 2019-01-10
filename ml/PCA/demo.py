
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

X = np.load('embeddings.npy')
pca = PCA(n_components=2)
X_r = pca.fit(X).transform(X)

print('explained variance ratio (first two components): {}'
      .format(pca.explained_variance_ratio_))
print('explained variance (first two components): {}'
      .format(pca.explained_variance_))
print('singular values (first two components): {}'
      .format(pca.explained_variance_))
print('features mean: {}'.format(pca.mean_))

plt.figure()
plt.scatter(X_r[:, 0], X_r[:, 1])
plt.show()
