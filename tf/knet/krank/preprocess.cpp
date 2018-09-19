
#include "feature_manager.h"

#include <iostream>

int main(int argc, char *argv[]) {
  if (argc < 5) {
    std::cout << "Usage: <data-file> <min-count> <save-file> <dict-file>" << std::endl;
    exit(-1);
  }
  krank::FeatureManager feature_manager(std::stoi(argv[2]));
  feature_manager.ReadFromFile(argv[1]);
  feature_manager.save(argv[3]);
  feature_manager.dump_rowkeys(argv[4]);

  return 0;
}
