
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <string>

#include "fasttext_api.h"

std::vector<std::string> LoadTags(const std::string &tagdictfile) {
  std::vector<std::string> tags;
  std::ifstream ifs(tagdictfile);
  if (!ifs.is_open()) {
    std::cerr << "open tag dict file [" << tagdictfile << "] failed."
              << std::endl;
    exit(-1);
  }

  std::string line;
  while (!ifs.eof()) {
    std::getline(ifs, line);
    tags.push_back(line);
  }
  return tags;
}

static void printUsage() {
  std::cerr
      << "usage: ./generate_synonyms <classifier-model> <tagdict> <output>"
      << std::endl;
}

// size of predictions must be at least 2
static std::string generate_key(
    const std::vector<std::pair<float, std::string>> &predictions) {
  std::string key;
  std::string label0 = predictions[0].second.substr(9);
  std::string label1 = predictions[1].second.substr(9);

  if (predictions[1].first < 0.01) {
    key = label0;
  } else {
    key = std::min(label0, label1) + '#' + std::max(label0, label1);
  }
  return key;
}

int main(int argc, char *argv[]) {
  if (argc < 4) {
    printUsage();
    exit(-1);
  }
  fasttext::FastTextApi classifier;

  classifier.LoadModel(argv[1]);
  auto tags = LoadTags(argv[2]);

  std::map<std::string, std::vector<std::string>> tagindex;
  for (auto &tagname : tags) {
    auto predictions = classifier.Predict(tagname, 2, false);
    if (predictions.size() < 2) {
      continue;
    }
    std::string key = generate_key(predictions);
    tagindex[key].push_back(tagname);
  }

  std::cout << "tagindex size: " << tagindex.size() << std::endl;

  // write tagindex to file
  std::ofstream ofs(argv[3], std::ios_base::out | std::ios_base::trunc);
  if (!ofs.is_open()) {
    std::cerr << "open output file [" << argv[3] << "] failed." << std::endl;
    exit(-1);
  }

  for (auto &p : tagindex) {
    ofs.write(p.first.data(), p.first.size());
    ofs.write(" ", 1);
    for (auto &tagname : p.second) {
      ofs.write("#", 1);
      ofs.write(tagname.data(), tagname.size());
    }
    ofs.write("\n", 1);
  }

  ofs.close();

  return 0;
}
