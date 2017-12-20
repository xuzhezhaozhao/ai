
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
    std::string key;
    if (predictions.size() == 2) {
      key = predictions[0].second + "#" + predictions[1].second;
    } else if (predictions.size() == 1) {
      key = predictions[0].second;
    } else {
      std::cerr << "no predictions returned, tagname = " << tagname
                << std::endl;
      continue;
    }
    tagindex[key].push_back(tagname);
  }

  // write to file

  return 0;
}
