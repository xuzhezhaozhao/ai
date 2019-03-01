#include "Clustering.h"
#include "IndexFlat.h"

#include <fstream>
#include <iostream>

namespace {

struct Args {
  std::string input;
  std::string output;
  int ncluster = -1;
  int min_points_per_centroid = 24;
  int max_points_per_centroid = 256;
  int niter = 20;
  int nredo = 1;
  int verbose = 1;
};

void print_help() {
  std::cerr
      << "Usage:\n"
      << "  -input                    input vector file path\n"
      << "  -output                   output file path\n"
      << "  -ncluster                 num of cluster\n"
      << "  -min_points_per_centroid  min points per centroid, default 24\n"
      << "  -max_points_per_centroid  max points per centroid, default 256\n"
      << "  -niter                    num of iter, default 20\n"
      << "  -nredo                    num of redo times, default 1\n"
      << "  -verbose                  verbose level, 0 or 1, default 1\n";
}

Args parse_args(int argc, char *argv[]) {
  Args args;
  for (int ai = 1; ai < argc; ai += 2) {
    std::string arg(argv[ai]);
    if (arg[0] != '-') {
      std::cerr << "Provided argument without a dash!" << std::endl;
      print_help();
      exit(EXIT_FAILURE);
    }
    if (arg == "-h") {
      std::cerr << "Here is the help! Usage:" << std::endl;
      print_help();
      exit(EXIT_FAILURE);
    } else if (arg == "-input") {
      args.input = std::string(argv[ai + 1]);
    } else if (arg == "-output") {
      args.output = std::string(argv[ai + 1]);
    } else if (arg == "-ncluster") {
      args.ncluster = std::stoi(argv[ai + 1]);
    } else if (arg == "-min_points_per_centroid") {
      args.min_points_per_centroid = std::stoi(argv[ai + 1]);
    } else if (arg == "-max_points_per_centroid") {
      args.max_points_per_centroid = std::stoi(argv[ai + 1]);
    } else if (arg == "-niter") {
      args.niter = std::stoi(argv[ai + 1]);
    } else if (arg == "-nredo") {
      args.nredo = std::stoi(argv[ai + 1]);
    } else if (arg == "-verbose") {
      args.verbose = std::stoi(argv[ai + 1]);
    } else {
      std::cerr << "Unknown argument: " << arg << std::endl;
      print_help();
      exit(EXIT_FAILURE);
    }
  }
  if (args.input.empty() || args.output.empty()) {
    std::cerr << "Empty input or output path." << std::endl;
    print_help();
    exit(EXIT_FAILURE);
  }
  if (args.ncluster < 0) {
    std::cerr << "<ncluster> should be larger than 0." << std::endl;
    print_help();
    exit(EXIT_FAILURE);
  }

  return args;
}
std::vector<std::string> split(const std::string &s,
                               const std::string &sep = " \t") {
  std::vector<std::string> result;
  size_t pos1 = 0;
  size_t pos2 = s.find_first_of(sep);
  while (std::string::npos != pos2) {
    result.push_back(s.substr(pos1, pos2 - pos1));
    pos1 = pos2 + 1;
    pos2 = s.find_first_of(sep, pos1);
  }
  if (pos1 != s.length()) {
    result.push_back(s.substr(pos1));
  }
  return result;
}

void load_input_vectors(const std::string &input, std::vector<float> &data,
                        std::vector<std::string> &keys, int64_t &ntotal,
                        int64_t &dim) {
  std::cerr << "Loading input vectors ..." << std::endl;
  std::ifstream ifs(input);
  if (!ifs.is_open()) {
    std::cerr << "Open file '" << input << "' failed." << std::endl;
    exit(EXIT_FAILURE);
  }
  std::string line;
  std::getline(ifs, line);
  auto tokens = split(line);
  if (tokens.size() != 2) {
    std::cerr << "Input error: Input first line should be '<ntotal> <dim>'"
              << std::endl;
    exit(EXIT_FAILURE);
  }
  try {
    ntotal = std::stoll(tokens[0]);
    dim = std::stoll(tokens[1]);
  } catch (const std::exception &e) {
    std::cerr << "Input error: Input first line should be '<ntotal> <dim>'"
              << std::endl;
    exit(EXIT_FAILURE);
  }
  std::cerr << "ntotal = " << ntotal << std::endl;
  std::cerr << "dim = " << dim << std::endl;
  data.clear();
  keys.clear();
  int64_t lineindex = 0;
  while (!ifs.eof()) {
    std::getline(ifs, line);
    ++lineindex;
    if (lineindex % 200000 == 0) {
      std::cerr << "load " << lineindex << " lines ..." << std::endl;
    }
    if (line == "") {
      break;
    }
    auto tokens = split(line);
    if (tokens.size() != dim + 1) {
      std::cerr << "Input error [1] in line " << lineindex << std::endl;
      exit(EXIT_FAILURE);
    }
    keys.push_back(tokens[0]);
    for (int64_t i = 1; i < dim + 1; ++i) {
      float v = 0.0;
      try {
        v = std::stof(tokens[i]);
      } catch (const std::exception &e) {
        std::cerr << "Input error [2] in line " << lineindex << std::endl;
        exit(EXIT_FAILURE);
      }
      data.push_back(v);
    }
  }
  ifs.close();
  if (data.size() != ntotal * dim) {
    std::cerr << "Input error [3]." << std::endl;
    exit(EXIT_FAILURE);
  }
  std::cerr << "Loading input vectors done" << std::endl;
}
}

int main(int argc, char *argv[]) {
  Args args = parse_args(argc, argv);

  // load input vectors
  std::vector<float> data;
  std::vector<std::string> keys;
  int64_t ntotal, dim;
  load_input_vectors(args.input, data, keys, ntotal, dim);

  // do kmeans
  std::cerr << "Do kmeans ..." << std::endl;
  std::vector<float> centroids(args.ncluster * dim);
  float error = faiss::kmeans_clustering(
      dim, ntotal, args.ncluster, data.data(), centroids.data(),
      args.min_points_per_centroid, args.max_points_per_centroid, args.niter,
      args.nredo, args.verbose);
  std::vector<faiss::Index::idx_t> assign(ntotal);
  std::vector<float> dist(ntotal);
  faiss::IndexFlatL2 index(dim);
  index.add(args.ncluster, centroids.data());
  index.search(ntotal, data.data(), 1, dist.data(), assign.data());
  std::cerr << "kmeans error = " << error << std::endl;
  std::cerr << "kmeans done" << std::endl;

  // output
  std::ofstream ofs(args.output);
  if (!ofs.is_open()) {
    std::cerr << "Open file '" << args.output << "' failed." << std::endl;
    exit(EXIT_FAILURE);
  }
  std::string head = "key centroid distance\n";
  ofs.write(head.data(), head.size());
  for (int64_t i = 0; i < ntotal; ++i) {
    // write key
    ofs.write(keys[i].data(), keys[i].size());
    ofs.write(" ", 1);

    // write centroid number
    auto s = std::to_string(assign[i]);
    ofs.write(s.data(), s.size());
    ofs.write(" ", 1);

    // write distance
    s = std::to_string(dist[i]);
    ofs.write(s.data(), s.size());
    ofs.write("\n", 1);
  }
  return 0;
}
