
#include "../Clustering.h"

#include <vector>
#include <iostream>

int main(int argc, char *argv[]) {
  int d = 3;
  int n = 10;
  int k = 2;

  std::vector<float> x(d * n);  // d * n
  std::vector<float> centroids(k * d);

  float mse = faiss::kmeans_clustering(d, n, k, x.data(), centroids.data());
  std::cout << "mse: " << mse << std::endl;

  return 0;
}
