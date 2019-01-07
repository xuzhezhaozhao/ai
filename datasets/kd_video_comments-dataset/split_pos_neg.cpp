#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <set>
#include <string>
#include <vector>

void inline to_lower(std::string &s) {
  std::transform(s.begin(), s.end(), s.begin(), ::tolower);
}

int main(int argc, char *argv[]) {
  if (argc != 5) {
    std::cout << "Usage: <neg_dict> <input> <pos_file> <neg_file>" << std::endl;
    exit(-1);
  }

  std::ifstream ifs_dict(argv[1]);
  assert(ifs_dict.is_open());

  std::ifstream ifs_input(argv[2]);
  assert(ifs_input.is_open());

  std::ofstream ofs_pos(argv[3]);
  assert(ofs_pos.is_open());

  std::ofstream ofs_neg(argv[4]);
  assert(ofs_neg.is_open());

  std::set<std::string> dict;
  std::string line;
  while (!ifs_dict.eof()) {
    std::getline(ifs_dict, line);
    if (line != "") {
      to_lower(line);
      std::cout << line << std::endl;
      dict.insert(line);
    }
  }
  std::vector<std::string> vdict(dict.begin(), dict.end());

  int64_t processed = 0;
  int64_t start = time(NULL);
  while (!ifs_input.eof()) {
    std::getline(ifs_input, line);
    to_lower(line);
    bool neg = false;
    for (auto &word : vdict) {
      if (strstr(line.data(), word.data()) != NULL) {
        // faster than line.find(), 9s vs 14s
        neg = true;
        break;
      }
    }
    if (neg) {
      ofs_neg.write(line.data(), line.size());
      ofs_neg.write("\n", 1);
    } else {
      ofs_pos.write(line.data(), line.size());
      ofs_pos.write("\n", 1);
    }
    ++processed;
    if (processed % 1000000 == 0) {
      std::cout << processed << " lines processed, elapsed "
                << time(NULL) - start << " s" << std::endl;
      start = time(NULL);
    }
  }
  return 0;
}
