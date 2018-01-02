
#include <assert.h>
#include <gflags/gflags.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>

DEFINE_string(raw_input, "", "raw input data, user pv from tdw");
DEFINE_bool(with_header, true, "raw input data with header");
DEFINE_bool(only_video, true, "only user video pv, exclude article pv.");
DEFINE_int32(interval, 1000000, "interval steps to print info");

DEFINE_string(output_user_watched_file, "user_watched.out",
              "output user watched file");

DEFINE_string(output_user_watched_ratio_file, "user_watched_ratio.out",
              "output user watched time ratio file for PCTR");
DEFINE_string(output_video_play_ratio_file, "", "used for similar videos");
DEFINE_double(video_play_ratio_bias, 0.0, "");

DEFINE_int32(user_min_watched, 20, "");
// TODO 截断至此大小
DEFINE_int32(user_max_watched, 512, "");

DEFINE_double(user_effective_watched_time_thr, 20.0, "");
DEFINE_double(user_effective_watched_ratio_thr, 0.3, "");

// 超过这个值的记录直接丢弃
DEFINE_int32(user_abnormal_watched_thr, 2048, "");

DEFINE_int32(supress_hot_arg1, 2, "");
DEFINE_int32(supress_hot_arg2, 1, "");

DEFINE_int32(threads, 1, "");

// 出现次数小于该值则丢弃
DEFINE_int32(min_count, 0, "");

DEFINE_string(ban_algo_ids, "", ", sep");
DEFINE_double(ban_algo_watched_ratio_thr, 0, "");

static std::vector<std::string> split(const std::string &s, char sep) {
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
  google::ParseCommandLineFlags(&argc, &argv, false);
  srand((uint32_t)time(NULL));

  // open raw input
  std::ifstream ifs(FLAGS_raw_input);
  if (!ifs.is_open()) {
    std::cerr << "open raw input data file [" << FLAGS_raw_input << "] failed."
              << std::endl;
    exit(-1);
  }

  std::map<uint64_t, std::vector<std::pair<int, float>>>
      histories;                        // uin -> {rowkey, watch_ratio}
  std::map<std::string, int> id2int;    // rowkey 到 index
  std::vector<std::string> ids;         // index 到 rowkey
  std::map<int, uint32_t> rowkeycount;  // 统计 rowkey 出现的次数
  std::map<int, std::pair<double, double>> video_play_ratios;

  // generate ban algo ids
  std::set<int> ban_algo_ids;
  if (!FLAGS_ban_algo_ids.empty()) {
    auto algo_ids = split(FLAGS_ban_algo_ids, ',');
    for (auto &algo_id : algo_ids) {
      ban_algo_ids.insert(std::stoi(algo_id));
    }
  }

  std::cout << "ban size: " << ban_algo_ids.size() << std::endl;

  int64_t lineprocessed = 0;
  int ndirty = 0;
  uint64_t total = 0;

  std::string line;
  while (!ifs.eof()) {
    std::getline(ifs, line);
    ++lineprocessed;
    auto tokens = split(line, ',');
    unsigned long uin = 0;
    int isvideo = 0;
    const std::string &rowkey = tokens[2];
    double video_duration = 0.0;
    double watched_time = 0.0;
    int algo_id = -123456;
    if (tokens.size() < 9) {
      ++ndirty;
      continue;
    }
    try {
      isvideo = std::stoi(tokens[3]);
      uin = std::stoul(tokens[1]);
      if (isvideo) {
        video_duration = std::stod(tokens[4]);
        watched_time = std::stod(tokens[5]);
        algo_id = std::stoi(tokens[8]);
      }
    } catch (const std::exception &e) {
      ++ndirty;
      continue;
    }

    if (!isvideo && FLAGS_only_video) {
      // 过滤图文
      continue;
    }

    if (id2int.find(rowkey) == id2int.end()) {
      id2int[rowkey] = static_cast<int>(ids.size());
      ids.push_back(rowkey);
    }
    int id = id2int[rowkey];

    double r = 0.0;
    if (isvideo && video_duration != 0.0) {
      // 统计视频播放比率
      video_play_ratios[id].first += watched_time;
      video_play_ratios[id].second += video_duration + FLAGS_video_play_ratio_bias;

      // 过滤出有效观看视频
      r = watched_time / video_duration;
      if (watched_time < FLAGS_user_effective_watched_time_thr &&
          r < FLAGS_user_effective_watched_ratio_thr) {
        continue;
      }
    }

    if (ban_algo_ids.find(algo_id) != ban_algo_ids.end()) {
      // 播放比必须大于一个阈值才不过滤
      if (r < FLAGS_ban_algo_watched_ratio_thr) {
        continue;
      }
    }

    if (!histories[uin].empty() && histories[uin].back().first == id) {
      // duplicate watched or error reported
      continue;
    }

    histories[uin].push_back({id, r});
    ++rowkeycount[id];
    ++total;

    if (lineprocessed % FLAGS_interval == 0) {
      std::cerr << lineprocessed << " lines processed." << std::endl;
    }
  }

  std::cerr << "user number: " << histories.size() << std::endl;
  std::cerr << "dirty lines number: " << ndirty << std::endl;
  std::cerr << "write user watched to file ..." << std::endl;

  std::ofstream ofs(FLAGS_output_user_watched_file);
  std::ofstream ofs_ratio(FLAGS_output_user_watched_ratio_file);
  std::ofstream ofs_video_play_ratio(FLAGS_output_video_play_ratio_file);
  if (!ofs.is_open()) {
    std::cerr << "open output user watched file failed. filename = "
              << FLAGS_output_user_watched_file << std::endl;
    exit(-1);
  }

  if (!ofs_ratio.is_open()) {
    std::cerr << "open output user watched ratio file failed. filename = "
              << FLAGS_output_user_watched_ratio_file << std::endl;
    exit(-1);
  }

  if (!ofs_video_play_ratio.is_open()) {
    std::cerr << "open output video play ratio file failed. filename = "
              << FLAGS_output_video_play_ratio_file << std::endl;
    exit(-1);
  }

  double mean_freq = 1.0 / rowkeycount.size();
  std::cerr << "mean_freq = " << mean_freq << std::endl;

  int noverfrep = 0;
  std::vector<std::pair<int, float>> valid;
  size_t i = 0;
  size_t total_valid = 0;
  for (auto &p : histories) {
    valid.clear();
    for (auto &x : p.second) {
      int id = x.first;

      assert((size_t)id < ids.size());

      if (rowkeycount[id] < FLAGS_min_count) {
        continue;
      }

      // 打压热门视频
      double freq = static_cast<double>(rowkeycount[id]) / total;
      if (FLAGS_supress_hot_arg1 != -1 &&
          freq > FLAGS_supress_hot_arg1 * mean_freq) {
        ++noverfrep;
        double r = rand() / static_cast<double>(RAND_MAX);
        if (r > (FLAGS_supress_hot_arg2 * mean_freq / freq)) {
          continue;
        }
      }
      valid.push_back(x);
    }

    if (valid.size() < FLAGS_user_min_watched ||
        valid.size() > FLAGS_user_abnormal_watched_thr) {
      continue;
    }

    total_valid += valid.size();
    auto suin = std::to_string(p.first);
    suin = "__label__" + suin;
    ofs.write(suin.data(), suin.size());
    ofs.write(" ", 1);

    ofs_ratio.write(suin.data(), suin.size());
    ofs_ratio.write(" ", 1);

    size_t j = 0;
    for (auto &x : valid) {
      int id = x.first;
      auto &rowkey = ids[id];
      ofs.write(rowkey.data(), rowkey.size());
      auto sr = std::to_string(x.second);
      ofs_ratio.write(sr.data(), sr.size());

      if (j != p.second.size() - 1) {
        ofs.write(" ", 1);
        ofs_ratio.write(" ", 1);
      }
      ++j;
    }

    if (i != histories.size() - 1) {
      ofs.write("\n", 1);
      ofs_ratio.write("\n", 1);
    }
    ++i;

    if (i % FLAGS_interval == 0) {
      std::cerr << i << " user's watched have been writedn." << std::endl;
    }
  }

  std::cerr << "write video play ratios, size = " << video_play_ratios.size()
            << std::endl;

  for (auto &p : video_play_ratios) {
    auto &rowkey = ids[p.first];
    ofs_video_play_ratio.write(rowkey.data(), rowkey.size());

    double total_watched_time = p.second.first;
    double total_duration = p.second.second;

    double ratio = 0;
    if (total_duration != 0.0) {
      ratio = total_watched_time / total_duration;
    }

    auto sratio = std::to_string(ratio);
    ofs_video_play_ratio.write(" ", 1);
    ofs_video_play_ratio.write(sratio.data(), sratio.size());
    ofs_video_play_ratio.write("\n", 1);
  }

  std::cerr << "noverfrep: " << noverfrep << std::endl;
  std::cerr << "total valid: " << total_valid << std::endl;

  ofs.close();
  ofs_ratio.close();
  ofs_video_play_ratio.close();

  return 0;
}