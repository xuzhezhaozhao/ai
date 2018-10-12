#ifndef _KNET_KRANK_FEATURE_MANAGER_H_
#define _KNET_KRANK_FEATURE_MANAGER_H_

#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "str_util.h"
#include "stringpiece.h"

#include "../cppml/src/transformer.h"

namespace fe {

struct UserAction {
  UserAction(float rinfo1_, float rinfo2_, const std::string& rowkey_,
             const std::string& first_video_rowkey_)
      : rinfo1(rinfo1_),
        rinfo2(rinfo2_),
        rowkey(rowkey_),
        first_video_rowkey(first_video_rowkey_) {}

  float rinfo1;
  float rinfo2;
  std::string rowkey;
  std::string first_video_rowkey;
};

struct TransformedUserAction {
  TransformedUserAction(int id_, int first_video_id_, float label_,
                        bool is_positive_, bool is_negative_)
      : id(id_),
        first_video_id(first_video_id_),
        label(label_),
        is_positive(is_positive_),
        is_negative(is_negative_) {}
  int id;
  int first_video_id;
  float label;
  bool is_positive;
  bool is_negative;
};

struct RawFeature {
  std::vector<UserAction> actions;
};

struct TransformedFeature {
  std::vector<TransformedUserAction> actions;
};

class FeaturePipline {
 public:
  FeaturePipline(int min_count = 0, float positive_threshold = 0.49f,
                 float negative_threshold = 0.04f,
                 float video_duration_biases = 1.0f,
                 bool use_smooth_label = false)
      : rowkey_indexer_(min_count, cppml::MinCountStringIndexer::MODE::COUNTER),
        processed_lines_(0),
        positive_threshold_(positive_threshold),
        negative_threshold_(negative_threshold),
        video_duration_biases_(video_duration_biases),
        use_smooth_label_(use_smooth_label) {}

  void feed(const std::string& line) {
    ++processed_lines_;
    if (processed_lines_ % 50000 == 0) {
      std::cerr << "[FeaturePipline] " << processed_lines_
                << " lines processed." << std::endl;
    }
    rowkey_indexer_.feed(line);
  }

  StringPiece GetRowkeyListToken(
      const std::vector<StringPiece>& features) const {
    return features[1];
  }

  void feed_end() { rowkey_indexer_.feed_end(); }

  TransformedFeature transform(const std::string& line) const {
    std::vector<StringPiece> pieces = Split(line, '\t');
    std::vector<StringPiece> h = Split(GetRowkeyListToken(pieces), ' ');
    TransformedFeature transformed_feature;
    for (auto& s : h) {
      // tokens: rowkey, first_video_rowkey, duration(ratio),
      // watch_time(stay_time)
      std::vector<StringPiece> tokens = Split(s, ':');

      // TODO(zhezhaoxu) Validate data
      int id = rowkey_indexer_.transform(std::string(tokens[0])).as_integer();
      int first_video_id =
          rowkey_indexer_.transform(std::string(tokens[1])).as_integer();
      float rinfo1 = std::stof(std::string(tokens[2]));
      float rinfo2 = std::stof(std::string(tokens[3]));
      float label = GetLabel(rinfo1, rinfo2);
      bool is_positive = IsPositive(rinfo1, rinfo2, label);
      bool is_negative = IsNegative(rinfo1, rinfo2, label);

      TransformedUserAction action(id, first_video_id, label, is_positive,
                                   is_negative);

      transformed_feature.actions.push_back(action);
    }
    return transformed_feature;
  }

  float GetLabel(float rinfo1, float rinfo2) const {
    float label = 0.0;
    if (use_smooth_label_) {
      label = std::min(1.0f, rinfo2 / (rinfo1 + video_duration_biases_));
    } else {
      // video effective play
      bool o1 = (rinfo1 < 20 && rinfo2 > rinfo1 * 0.8);
      bool o2 = (rinfo2 >= 20 || rinfo2 > rinfo1 * 0.8);
      label = (o1 || o2) ? 1.0 : 0.0;
    }
    return label;
  }

  bool IsPositive(float rinfo1, float rinfo2, float label) const {
    if (rinfo2 >= rinfo1 * positive_threshold_ || rinfo2 >= 20.0) {
      return true;
    }
    return false;
  }

  bool IsNegative(float rinfo1, float rinfo2, float label) const {
    if (label < negative_threshold_ || rinfo2 < 5) {
      return true;
    }
    return false;
  }

  TransformedFeature transform(const RawFeature& feature) const {
    TransformedFeature transformed_feature;
    for (auto& action : feature.actions) {
      int id = rowkey_indexer_.transform(action.rowkey).as_integer();
      int first_video_id =
          rowkey_indexer_.transform(action.first_video_rowkey).as_integer();
      float label = GetLabel(action.rinfo1, action.rinfo2);
      bool is_positive = IsPositive(action.rinfo1, action.rinfo2, label);
      bool is_negative = IsNegative(action.rinfo1, action.rinfo2, label);
      transformed_feature.actions.push_back(
          {id, first_video_id, label, is_positive, is_negative});
    }
    return transformed_feature;
  }

  int getRowkeyId(const std::string& rowkey) const {
    return rowkey_indexer_.transform(rowkey).as_integer();
  }

  FeaturePipline& setPositiveThreshold(float thr) {
    this->positive_threshold_ = thr;
    return *this;
  }

  FeaturePipline& setNegativeThreshold(float thr) {
    this->negative_threshold_ = thr;
    return *this;
  }

  void save(std::ofstream& out) const {
    rowkey_indexer_.save(out);
    out.write((char*)&positive_threshold_, sizeof(float));
    out.write((char*)&negative_threshold_, sizeof(float));
    out.write((char*)&video_duration_biases_, sizeof(float));
    out.write((char*)&use_smooth_label_, sizeof(bool));
  }
  void load(std::istream& in) {
    rowkey_indexer_.load(in);
    in.read((char*)&positive_threshold_, sizeof(float));
    in.read((char*)&negative_threshold_, sizeof(float));
    in.read((char*)&video_duration_biases_, sizeof(float));
    in.read((char*)&use_smooth_label_, sizeof(bool));
    std::cout << "[FeaturePipline] positive_threhold = " << positive_threshold_
              << std::endl;
    std::cout << "[FeaturePipline] negative_threhold = " << negative_threshold_
              << std::endl;
    std::cout << "[FeaturePipline] video_duration_biases = "
              << video_duration_biases_ << std::endl;
    std::cout << "[FeaturePipline] use_smooth_label = " << use_smooth_label_
              << std::endl;
  }
  void dump_rowkeys(const std::string& filename) const {
    rowkey_indexer_.dump(filename);
  }

 private:
  cppml::MinCountStringIndexer rowkey_indexer_;
  int64_t processed_lines_;
  float positive_threshold_;
  float negative_threshold_;
  float video_duration_biases_;
  bool use_smooth_label_;
};

class FeatureManager {
 public:
  FeatureManager(int min_count = 1, float positive_threshold = 0.8,
                 float negative_threshold = 0.1f,
                 float video_duration_biases = 1.0f,
                 bool use_smooth_label = false)
      : feature_pipline_(min_count, positive_threshold, negative_threshold,
                         video_duration_biases, use_smooth_label) {}

  void ReadFromFiles(const std::string& rowkey_count_file) {
    std::ifstream ifs(rowkey_count_file);
    if (!ifs.is_open()) {
      std::cerr << "Open " << rowkey_count_file << " failed." << std::endl;
      exit(-1);
    }

    std::string line;
    while (!ifs.eof()) {
      std::getline(ifs, line);
      if (line.empty()) {
        continue;
      }
      feature_pipline_.feed(line);
    }
    feature_pipline_.feed_end();
  }

  void save(const std::string& filename) const {
    std::ofstream ofs(filename);
    if (!ofs.is_open()) {
      std::cerr << "Save FeatureManager failed (open file '" << filename
                << "' failed)." << std::endl;
      exit(-1);
    }
    feature_pipline_.save(ofs);
  }
  void load(const std::string& filename) {
    std::ifstream ifs(filename, std::ios::in & std::ios::binary);
    if (!ifs.is_open()) {
      std::cerr << "Load FeatureManager failed (open file failed)."
                << std::endl;
      exit(-1);
    }
    feature_pipline_.load(ifs);
  }

  void load_from_string(const std::string& fm) {
    std::istringstream ss(fm);
    feature_pipline_.load(ss);
  }

  void dump_rowkeys(const std::string& filename) const {
    feature_pipline_.dump_rowkeys(filename);
  }

  TransformedFeature transform(const std::string& line) const {
    return feature_pipline_.transform(line);
  }

  TransformedFeature transform(const RawFeature& feature) const {
    return feature_pipline_.transform(feature);
  }

  int getRowkeyId(const std::string& rowkey) const {
    return feature_pipline_.getRowkeyId(rowkey);
  }

 private:
  FeaturePipline feature_pipline_;
};

}  // namespace fe

#endif
