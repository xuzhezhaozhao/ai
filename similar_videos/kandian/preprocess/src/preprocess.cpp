
#include <gflags/gflags.h>
#include <assert.h>

#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

DEFINE_string(raw_input, "", "raw input data, user pv from tdw");
DEFINE_bool(with_header, true, "raw input data with header");
DEFINE_bool(only_video, true, "only user video pv, exclude article pv.");
DEFINE_int32(interval, 1000000, "interval steps to print info");

DEFINE_string(output_user_watched_file, "user_watched.out",
              "output user watched file");

DEFINE_int32(user_min_watched, 20, "");
// 截断至此大小
DEFINE_int32(user_max_watched, 512, "");

// 超过这个值的记录直接丢弃
DEFINE_int32(user_abnormal_watched_thr, 2048, "");

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
  int ndirty = 0;
  while (!ifs.eof()) {
    std::getline(ifs, line);
    ++lineprocessed;
    if (FLAGS_with_header && lineprocessed == 1) {
      continue;
    }
    if (line.empty()) {
      continue;
    }
    auto tokens = split(line, ",");
    if (tokens.size() < 6) {
      std::cerr << "tokens size is less than 6. line number " << lineprocessed
                << std::endl;
      ++ndirty;
      continue;
    }
    bool isempty = false;
    for (int i = 0; i < 6; ++i) {
      if (tokens[i] == "") {
        isempty = true;
        break;
      }
    }
    if (isempty) {
      //std::cerr << "token is empty. lineprocessed = " << lineprocessed
                //<< ", line = " << line << std::endl;

      ++ndirty;
      continue;
    }

    unsigned long uin = std::stoul(tokens[1]);
    const std::string &rowkey = tokens[2];
    int isvideo = std::stoi(tokens[3]);

    if (!isvideo && FLAGS_only_video) {
      continue;
    }

    if (id2int.find(rowkey) == id2int.end()) {
      id2int[rowkey] = static_cast<int>(ids.size());
      ids.push_back(rowkey);
    }
    int id = id2int[rowkey];

    // TODO(zhezhaoxu) supress hot
    histories[uin].push_back(id);

    if (lineprocessed % FLAGS_interval == 0) {
      std::cerr << lineprocessed << "lines processed.";
    }
  }

  std::cerr << "user number: " << histories.size() << std::endl;
  std::cerr << "dirty lines number: " << ndirty << std::endl;

  std::cerr << "write user watched to file ..." << std::endl;
  std::ofstream ofs(FLAGS_output_user_watched_file);
  if (!ofs.is_open()) {
    std::cerr << "open output user watched file failed. filename = "
              << FLAGS_output_user_watched_file << std::endl;
    exit(-1);
  }
  size_t i = 0;
  for (auto &p : histories) {
    auto suin = std::to_string(p.first);
    suin = "__label__" + suin;
    ofs.write(suin.data(), suin.size());
    ofs.write(" ", 1);
    size_t j = 0;
    for (auto &id : p.second) {
      assert((size_t)id < ids.size());
      auto &s = ids[id];
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

    if (i % FLAGS_interval == 0) {
      std::cerr << i << " user's watched have been writedn." << std::endl;
    }
  }

  return 0;
}
