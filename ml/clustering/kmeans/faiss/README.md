
Kmeans implementation from faiss (https://github.com/facebookresearch/faiss).

Also see ./tests/troubleshooting.md

## Performance (with openblas)
faiss kmeans: 574726 vector, 100 dimentions, 3000 clusters, min 100, max 800, iter 100.
耗时: 348 seconds
mse: 1.55629e+06

sklearn kmeans:574726 vector, 100 dimentions, 3000 clusters, iter 100, tol 1e-12.
耗时:
mse:
