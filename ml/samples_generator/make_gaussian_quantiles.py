
import matplotlib.pyplot as plt
from sklearn.datasets import make_gaussian_quantiles

X1, Y1 = make_gaussian_quantiles(
    n_samples=1000, n_features=2, n_classes=3, mean=[1, 2], cov=2)
plt.scatter(X1[:, 0], X1[:, 1], marker='o', c=Y1)
plt.show()
