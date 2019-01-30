#include "../Clustering.h"
#include "../IndexFlat.h"

#include <cassert>
#include <fstream>
#include <iostream>
#include <vector>

static std::vector<std::string> split(const std::string &s, char sep = ' ') {
  std::vector<std::string> result;

  size_t pos1 = 0;
  size_t pos2 = s.find(sep);
  while (std::string::npos != pos2) {
    result.push_back(s.substr(pos1, pos2 - pos1));
    pos1 = pos2 + 1;
    pos2 = s.find(sep, pos1);
  }
  if (pos1 != s.length()) {
    result.push_back(s.substr(pos1));
  }
  return result;
}

int main(int argc, char *argv[]) {
  if (argc < 3) {
    std::cout << "Usage: <vec> <output>" << std::endl;
    std::cout << "Note: <vec> file is fasttext vector format." << std::endl;
    exit(-1);
  }
  std::string filename = argv[1];
  std::string output = argv[2];
  std::ifstream ifs(filename);
  assert(ifs.is_open());
  std::string line;
  std::getline(ifs, line);
  auto tokens = split(line);
  assert(tokens.size() == 2);
  int ntotal = std::stoi(tokens[0]);
  int dim = std::stoi(tokens[1]);
  std::cout << "ntotal = " << ntotal << std::endl;
  std::cout << "dim = " << dim << std::endl;
  std::vector<float> data;
  std::vector<std::string> dict;
  while (!ifs.eof()) {
    std::getline(ifs, line);
    if (line == "") {
      break;
    }
    auto tokens = split(line);
    assert(tokens.size() == dim + 1);
    dict.push_back(tokens[0]);
    for (int i = 1; i < dim + 1; ++i) {
      data.push_back(std::stof(tokens[i]));
    }
  }
  ifs.close();
  assert(data.size() == ntotal * dim);
  int k = 100;
  std::vector<float> centroids(k * dim);
  faiss::kmeans_clustering(dim, ntotal, k, data.data(), centroids.data(), 100,
                           2000, 100, 5, true);
  std::vector<faiss::Index::idx_t> assign(ntotal);
  std::vector<float> dis(ntotal);
  faiss::IndexFlatL2 index(dim);
  index.add(k, centroids.data());
  index.search(ntotal, data.data(), 1, dis.data(), assign.data());
  std::ofstream ofs(output);
  assert(ofs.is_open());
  for (int i = 0; i < ntotal; ++i) {
    auto s = std::to_string(assign[i]);
    ofs.write(s.data(), s.size());
    ofs.write(" ", 1);
    ofs.write(dict[i].data(), dict[i].size());
    ofs.write(" ", 1);
    s = std::to_string(dis[i]);
    ofs.write(s.data(), s.size());
    ofs.write("\n", 1);
  }
  ofs.close();

  return 0;
}
