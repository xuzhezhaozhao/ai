
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
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <vector>

DEFINE_string(raw_input, "raw_input", "raw input data, user pv from tdw");
DEFINE_bool(with_header, true, "raw input data with header");
DEFINE_bool(only_video, true, "only user video pv, exclude article pv.");
DEFINE_int32(interval, 1000000, "interval steps to print info");

DEFINE_string(output_user_watched_file, "output_user_watched_file",
    "output user watched file");

DEFINE_string(output_user_watched_ratio_file, "output_user_watched_ratio_file",
    "output user watched time ratio file for PCTR");

DEFINE_string(output_video_play_ratio_file, "output_video_play_ratio_file", "used for similar videos");

// 视频和图文的 rowkey 词典文件
DEFINE_string(output_video_dict_file, "output_video_dict_file", "");
DEFINE_string(output_article_dict_file, "output_article_dict_file", "");
DEFINE_string(output_video_click_file, "output_video_click_file", "");

DEFINE_double(video_play_ratio_bias, 0.0, "");

DEFINE_int32(user_min_watched, 20, "");

// TODO 截断至此大小
DEFINE_int32(user_max_watched, 512, "");

DEFINE_double(user_effective_watched_time_thr, 20.0, "");
DEFINE_double(user_effective_watched_ratio_thr, 0.3, "");

// 超过这个值的记录直接丢弃
DEFINE_int32(user_abnormal_watched_thr, 512, "");

DEFINE_int32(supress_hot_arg1, 2, "");
DEFINE_int32(supress_hot_arg2, 1, "");

DEFINE_int32(article_supress_hot_arg1, 2, "");
DEFINE_int32(article_supress_hot_arg2, 1, "");

DEFINE_int32(article_read_time_thr, 0, ""); // seconds
DEFINE_double(article_read_ratio_thr, 0.0, ""); // %

DEFINE_int32(threads, 1, "");

// 出现次数小于该值则丢弃
DEFINE_int32(min_count, 0, "");

DEFINE_string(ban_algo_ids, "", ", sep");
DEFINE_double(ban_algo_watched_ratio_thr, 0, "");

DEFINE_bool(ban_unknow_algo_id, false, "ban unknow algo id -1");

struct VideoInfo {
  double total_watched_time;
  double total_duration;

  uint64_t click_times;
  uint64_t effective_click_times;
};

// uin ==> {rowkey, watch_ratio}
static std::unordered_map<uint32_t, std::vector<std::pair<int, float>>> histories;

// will OOM
//static std::unordered_map<uint32_t, std::set<int>> histories_set;

static std::unordered_map<std::string, int> rowkey2int;    // rowkey 到 index
static std::vector<std::string> rowkeys;         // index 到 rowkey
static std::unordered_map<int, uint32_t> rowkeycount;  // 统计 rowkey 出现的次数
static std::unordered_map<int, VideoInfo> video_info;
static std::unordered_set<int> ban_algo_ids;
static uint64_t total = 0;
static std::unordered_set<int> all_videos;
static std::unordered_set<int> all_articles;

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

static void GenerateBanAlgoIds() {
  if (!FLAGS_ban_algo_ids.empty()) {
    auto algo_ids = split(FLAGS_ban_algo_ids, ',');
    for (auto &algo_id : algo_ids) {
      ban_algo_ids.insert(std::stoi(algo_id));
    }
  }
  std::cerr << "ban algo ids size: " << ban_algo_ids.size() << std::endl;
}

static bool IsContain(int id, const std::vector<std::pair<int, float>> &recoreds) {
  for (const auto &p : recoreds) {
    if (id == p.first) {
      return true;
    }
  }

  return false;
}

static void OpenFileRead(const std::string &file, std::ifstream &ifs) {
  ifs.open(file);
  if (!ifs.is_open()) {
    std::cerr << "open file [" << file << "] to read failed." << std::endl;
    exit(-1);
  }
}

static void OpenFileWrite(const std::string &file, std::ofstream &ofs) {
  ofs.open(file);
  if (!ofs.is_open()) {
    std::cerr << "open file [" << file << "] to write failed." << std::endl;
    exit(-1);
  }
}

static void ProcessRawInput() {
  std::ifstream ifs;
  OpenFileRead(FLAGS_raw_input, ifs);

  int64_t lineprocessed = 0;
  int ndirty = 0;
  int nalgoiderror = 0;

  std::string line;
  while (!ifs.eof()) {
    std::getline(ifs, line);
    ++lineprocessed;
    auto tokens = split(line, ',');
    if (tokens.size() < 9) {
      ++ndirty;
      continue;
    }
    uint32_t uin = 0;
    int isvideo = 0;
    const std::string &rowkey = tokens[2];
    if (rowkey.size() < 5) {
      ++ndirty;
      continue;
    }
    double video_duration = 0.0;
    double watched_time = 0.0;
    int algo_id = -123456;
    try {
      isvideo = std::stoi(tokens[3]);
      uin = (uint32_t)std::stoul(tokens[1]);
      video_duration = std::stod(tokens[4]);
      watched_time = std::stod(tokens[5]);
      algo_id = std::stoi(tokens[8]);
    } catch (const std::exception &e) {
      ++ndirty;
      continue;
    }

    if (!histories[uin].empty() &&
        histories[uin].size() > FLAGS_user_abnormal_watched_thr) {
      continue;
    }

    if (FLAGS_ban_unknow_algo_id && algo_id == -1) {
      ++nalgoiderror;
      continue;
    }

    if (!isvideo && FLAGS_only_video) {
      // 过滤图文
      continue;
    }

    if (rowkey2int.find(rowkey) == rowkey2int.end()) {
      rowkey2int[rowkey] = static_cast<int>(rowkeys.size());
      rowkeys.push_back(rowkey);
    }
    int id = rowkey2int[rowkey];

    double r = 0.0;
    if (isvideo) {
      if (video_duration > 0) {
        // 统计视频播放比率
        video_info[id].total_watched_time += watched_time;
        video_info[id].total_duration += video_duration + FLAGS_video_play_ratio_bias;
        video_info[id].click_times += 1;

        // 过滤出有效观看视频
        r = watched_time / video_duration;
        if (watched_time < FLAGS_user_effective_watched_time_thr &&
            r < FLAGS_user_effective_watched_ratio_thr) {
          continue;
        }
        if (ban_algo_ids.find(algo_id) != ban_algo_ids.end()) {
          // 播放比必须大于一个阈值才不过滤
          if (r < FLAGS_ban_algo_watched_ratio_thr) {
            continue;
          }
        }
        video_info[id].effective_click_times += 1;
        all_videos.insert(id);
      } else {
        ++ndirty;
        continue;
      }
    } else {
      // 过滤无效图文阅读
      if (watched_time < FLAGS_article_read_time_thr ||
          video_duration < FLAGS_article_read_ratio_thr) {
        continue;
      }
      all_articles.insert(id);
    }

    //if (!histories[uin].empty() && histories[uin].back().first == id) {
      //// duplicate watched or error reported
      //if (r > histories[uin].back().second) {
        //// update watched ratio
        //histories[uin].back().second = (float)r;
      //}
      //continue;
    //}

    if (IsContain(id, histories[uin])) {
      // duplicate watched, skip
      continue;
    }

    // OOM!!!
    //if (!histories_set[uin].empty() && histories_set[uin].count(id) > 0) {
      //// duplicate watched, skip
      //continue;
    //}

    histories[uin].push_back({id, r});
    //histories_set[uin].insert(id);
    ++rowkeycount[id];
    ++total;

    if (lineprocessed % FLAGS_interval == 0) {
      std::cerr << lineprocessed << " lines processed." << std::endl;
    }
  }

  std::cerr << "user number: " << histories.size() << std::endl;
  std::cerr << "dirty lines number: " << ndirty << std::endl;
  std::cerr << "algo id -1 lines number: " << nalgoiderror << std::endl;
  std::cerr << "write user watched to file ..." << std::endl;
}

static void WriteUserWatchedInfoFile() {
  std::ofstream ofs;
  std::ofstream ofs_ratio;
  OpenFileWrite(FLAGS_output_user_watched_file, ofs);
  OpenFileWrite(FLAGS_output_user_watched_ratio_file, ofs_ratio);

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

      assert((size_t)id < rowkeys.size());

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
      auto &rowkey = rowkeys[id];
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

  std::cerr << "noverfrep: " << noverfrep << std::endl;
  std::cerr << "total valid: " << total_valid << std::endl;

  ofs.close();
  ofs_ratio.close();
}

static void WriteVideoInfoFile() {
  std::ofstream ofs_video_play_ratio;
  std::ofstream ofs_click;
  OpenFileWrite(FLAGS_output_video_play_ratio_file, ofs_video_play_ratio);
  OpenFileWrite(FLAGS_output_video_click_file, ofs_click);

  std::cerr << "write video play ratios, size = " << video_info.size() << std::endl;

  for (auto &p : video_info) {
    auto &rowkey = rowkeys[p.first];
    ofs_video_play_ratio.write(rowkey.data(), rowkey.size());
    ofs_click.write(rowkey.data(), rowkey.size());

    double total_watched_time = p.second.total_watched_time;
    double total_duration = p.second.total_duration;

    double ratio = 0;
    if (total_duration != 0.0) {
      ratio = total_watched_time / total_duration;
    }

    auto sratio = std::to_string(ratio);
    ofs_video_play_ratio.write(" ", 1);
    ofs_video_play_ratio.write(sratio.data(), sratio.size());
    ofs_video_play_ratio.write("\n", 1);

    auto sclick_times = std::to_string(p.second.click_times);
    auto seffective_click_times = std::to_string(p.second.effective_click_times);
    ofs_click.write(" ", 1);
    ofs_click.write(sclick_times.data(), sclick_times.size());
    ofs_click.write(" ", 1);
    ofs_click.write(seffective_click_times.data(), seffective_click_times.size());
    ofs_click.write("\n", 1);
  }

  ofs_video_play_ratio.close();
  ofs_click.close();
}

static void WriteVideoDictFile() {
  if (FLAGS_output_video_dict_file.empty()) {
    return;
  }
  std::ofstream ofs;
  std::cerr << "videos dict size = " << all_videos.size() << std::endl;
  OpenFileWrite(FLAGS_output_video_dict_file, ofs);
  for (auto id : all_videos) {
    auto &rowkey = rowkeys[id];
    ofs.write(rowkey.data(), rowkey.size());
    ofs.write("\n", 1);
  }
  ofs.close();
}

static void WriteArticleDictFile() {
  if (FLAGS_output_article_dict_file.empty()) {
    return;
  }
  std::ofstream ofs;
  OpenFileWrite(FLAGS_output_article_dict_file, ofs);
  std::cerr << "articles dict size = " << all_articles.size() << std::endl;
  for (auto id : all_articles) {
    auto &rowkey = rowkeys[id];
    ofs.write(rowkey.data(), rowkey.size());
    ofs.write("\n", 1);
  }
  ofs.close();
}

int main(int argc, char *argv[]) {
  google::ParseCommandLineFlags(&argc, &argv, false);
  srand((uint32_t)time(NULL));

  GenerateBanAlgoIds();
  ProcessRawInput();

  WriteUserWatchedInfoFile();
  WriteVideoInfoFile();
  WriteVideoDictFile();
  WriteArticleDictFile();

  return 0;
}
