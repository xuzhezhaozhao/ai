#include "../Clustering.h"

#include <iostream>
#include <vector>

int main(int argc, char *argv[]) {
  (void) argc;
  (void) argv;

  int d = 3;
  int n = 10;
  int k = 2;

  std::vector<float> x(d * n);  // d * n
  std::vector<float> centroids(k * d);

  float mse = faiss::kmeans_clustering(d, n, k, x.data(), centroids.data(), 2,
                                       256, 100, 1, true);
  std::cout << "mse: " << mse << std::endl;

  return 0;
}
