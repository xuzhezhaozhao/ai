
#include <gflags/gflags.h>

#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

DEFINE_string(raw_input, "", "raw input data, user pv from tdw");
DEFINE_bool(only_video, true, "only user video pv, exclude article pv.");
DEFINE_int32(interval, 1000000, "interval steps to print info");

DEFINE_string(output_int2id_file, "int2id.out", "output int to id map file");
DEFINE_string(output_user_watched_file, "user_watched.out",
              "output user watched file");

static std::vector<std::string> split(const std::string &s,
                                      const std::string &delim) {
  std::vector<std::string> result;

  size_t pos1 = 0;
  size_t pos2 = s.find(delim);
  while (std::string::npos != pos2) {
    result.push_back(s.substr(pos1, pos2 - pos1));

    pos1 = pos2 + delim.size();
    pos2 = s.find(delim, pos1);
  }
  if (pos1 != s.length()) {
    result.push_back(s.substr(pos1));
  }
  return result;
}

int main(int argc, char *argv[]) {
  google::ParseCommandLineFlags(&argc, &argv, false);

  std::ifstream ifs(FLAGS_raw_input);
  if (!ifs.is_open()) {
    std::cerr << "open raw input data file [" << FLAGS_raw_input << "] failed."
              << std::endl;
    exit(-1);
  }

  std::map<unsigned long, std::vector<int>> histories;
  std::map<std::string, int> id2int;
  std::vector<std::string> ids;

  std::string line;
  int lineprocessed = 0;
  while (!ifs.eof()) {
    std::getline(ifs, line);
    ++lineprocessed;
    if (line.empty()) {
      continue;
    }
    auto tokens = split(line, ",");
    if (tokens.size() < 6) {
      std::cerr << "tokens size is less than 6. line number " << lineprocessed
                << std::endl;
      continue;
    }
    bool isempty = false;
    for (auto &token : tokens) {
      if (token == "") {
        isempty = true;
        break;
      }
    }
    if (isempty) {
      std::cerr << "token is empty." << std::endl;
      continue;
    }

    unsigned long uin = std::stoul(tokens[1]);
    const std::string &rowkey = tokens[2];
    int isvideo = std::stoi(tokens[3]);

    if (!isvideo && FLAGS_only_video) {
      continue;
    }

    if (id2int.find(rowkey) == id2int.end()) {
      id2int[rowkey] = static_cast<int>(id2int.size());
      ids.push_back(rowkey);
    }
    int id = id2int[rowkey];

    // TODO(zhezhaoxu) supress hot
    histories[uin].push_back(id);

    if (lineprocessed % FLAGS_interval == 0) {
      std::cerr << lineprocessed << "lines processed.";
    }
  }

  std::cerr << "write user watched to file ..." << std::endl;
  std::ofstream ofs(FLAGS_output_user_watched_file);
  size_t i = 0;
  for (auto &p : histories) {
    auto s = std::to_string(p.first);
    ofs.write(s.data(), s.size());
    ofs.write(" ", 1);
    size_t j = 0;
    for (auto &w : p.second) {
      auto s = std::to_string(w);
      ofs.write(s.data(), s.size());
      if (j != p.second.size() - 1) {
        ofs.write(" ", 1);
      }
      ++j;
    }
    if (i != histories.size() - 1) {
      ofs.write("\n", 1);
    }
    ++i;
  }

  std::cerr << "write int2id map to file ..." << std::endl;
  std::ofstream ofs2(FLAGS_output_int2id_file);
  for (size_t index = 0; index < ids.size(); ++index) {
    ofs2.write(ids[index].data(), ids[index].size());
    if (index != ids.size() - 1) {
      ofs2.write("\n", 1);
    }
  }

  return 0;
}
