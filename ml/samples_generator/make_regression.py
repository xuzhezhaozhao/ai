
import matplotlib.pyplot as plt

from sklearn.datasets.samples_generator import make_regression
X, y, coef = make_regression(n_samples=1000, n_features=1, noise=10, coef=True)
plt.scatter(X, y,  color='black')
plt.plot(X, X * coef, color='blue', linewidth=3)
plt.xticks(())
plt.yticks(())
plt.show()
