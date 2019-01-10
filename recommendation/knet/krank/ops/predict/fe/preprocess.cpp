#include "feature_manager.h"

#include <iostream>

int main(int argc, char *argv[]) {
  if (argc < 9) {
    std::cout
        << "Usage: <data-file> <min-count> <positive_threhold> "
        "<negative_threhold> <video_duration_biases> <use_smooth_label>"
        " <save-file> <dict-file>"
        << std::endl;
    exit(-1);
  }
  fe::FeatureManager feature_manager(std::stoi(argv[2]), std::stof(argv[3]),
                                     std::stof(argv[4]), std::stof(argv[5]),
                                     std::stoi(argv[6]));
  feature_manager.ReadFromFiles(argv[1]);
  feature_manager.save(argv[7]);
  feature_manager.dump_rowkeys(argv[8]);

  return 0;
}
