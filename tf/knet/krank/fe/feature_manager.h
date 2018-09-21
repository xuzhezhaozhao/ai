#ifndef _KNET_KRANK_FEATURE_MANAGER_H_
#define _KNET_KRANK_FEATURE_MANAGER_H_

#include <fstream>
#include <iostream>
#include <string>

#include "str_util.h"
#include "stringpiece.h"

#include "../../../../cppml/src/transformer.h"

namespace fe {

struct UserAction {
  UserAction(float rinfo1_, float rinfo2_, const std::string& rowkey_,
             bool isvideo_)
      : rinfo1(rinfo1_), rinfo2(rinfo2_), rowkey(rowkey_), isvideo(isvideo_) {}

  float rinfo1;
  float rinfo2;
  std::string rowkey;
  bool isvideo;
};

struct TransformedUserAction {
  TransformedUserAction(int id_, bool label_, bool unlike_, bool isvideo_)
      : id(id_), label(label_), unlike(unlike_), isvideo(isvideo_) {}
  int id;
  bool label;
  bool unlike;
  bool isvideo;
};

struct RawFeature {
  std::vector<UserAction> actions;
};

struct TransformedFeature {
  std::vector<TransformedUserAction> actions;
};

class FeaturePipline {
 public:
  FeaturePipline(int min_count = 0)
      : rowkey_indexer_(min_count, cppml::MinCountStringIndexer::MODE::COUNTER),
        processed_lines_(0) {}

  void feed(const std::string& line) {
    ++processed_lines_;
    if (processed_lines_ % 50000 == 0) {
      std::cerr << "[FeaturePipline] " << processed_lines_
                << " lines processed." << std::endl;
    }

    std::vector<StringPiece> features = Split(line, '\t');
    // features[0]: uin
    // features[1]: rowkey list

    if (features.size() != 2) {
      std::cerr << "line format error. line = " << line << std::endl;
      return;
    }
    std::vector<StringPiece> h = Split(GetRowkeyListToken(features), ' ');
    ParseRowkeyList(h);
  }

  StringPiece GetRowkeyListToken(
      const std::vector<StringPiece>& features) const {
    return features[1];
  }

  void ParseRowkeyList(const std::vector<StringPiece>& h) {
    for (auto& s : h) {
      std::vector<StringPiece> tokens = Split(s, ':');
      // tokens: rowkey, isvideo, duration(ratio), watch_time(stay_time)
      if (tokens.size() != 4) {
        std::cerr << "rowkey list format error. tokens.size = " << tokens.size()
                  << ", piece = " << s << std::endl;
        continue;
      }
      rowkey_indexer_.feed(std::string(tokens[0]));
    }
  }
  void feed_end() { rowkey_indexer_.feed_end(); }

  TransformedFeature transform(const std::string& line) const {
    std::vector<StringPiece> pieces = Split(line, '\t');
    std::vector<StringPiece> h = Split(GetRowkeyListToken(pieces), ' ');
    TransformedFeature transformed_feature;
    for (auto& s : h) {
      // tokens: rowkey, isvideo, duration(ratio), watch_time(stay_time)
      std::vector<StringPiece> tokens = Split(s, ':');

      // TODO(zhezhaoxu) Validate data
      int id = rowkey_indexer_.transform(std::string(tokens[0])).as_integer();
      bool isvideo = (tokens[1][0] == '1');
      float rinfo1 = std::stof(std::string(tokens[2]));
      float rinfo2 = std::stof(std::string(tokens[3]));
      bool label = GetLabel(isvideo, rinfo1, rinfo2);
      bool unlike = GetUnlike(isvideo, rinfo1, rinfo2);

      TransformedUserAction action(id, label, unlike, isvideo);

      transformed_feature.actions.push_back(action);
    }
    return transformed_feature;
  }

  bool GetLabel(bool isvideo, float rinfo1, float rinfo2) const {
    bool label = false;
    if (isvideo) {
      // video effective play
      bool o1 = (rinfo1 < 30 && rinfo2 > rinfo1 * 0.8);
      bool o2 = (rinfo1 >= 30 && rinfo2 > rinfo1 * 0.5);
      label = o1 || o2;
    } else {
      // article effective reading
      label = (rinfo1 > 0.9 || rinfo2 > 40);
    }
    return label;
  }

  bool GetUnlike(bool isvideo, float rinfo1, float rinfo2) const {
    bool unlike = false;
    if (isvideo) {
      unlike = (rinfo2 <= 4);
    } else {
      unlike = (rinfo2 <= 5);
    }
    return unlike;
  }

  TransformedFeature transform(const RawFeature& feature) const {
    TransformedFeature transformed_feature;
    for (auto& action : feature.actions) {
      int id = rowkey_indexer_.transform(action.rowkey).as_integer();
      bool label = GetLabel(action.isvideo, action.rinfo1, action.rinfo2);
      bool unlike = GetUnlike(action.isvideo, action.rinfo1, action.rinfo2);
      transformed_feature.actions.push_back(
          {id, label, unlike, action.isvideo});
    }
    return transformed_feature;
  }

  void save(std::ofstream& out) const { rowkey_indexer_.save(out); }
  void load(std::ifstream& in) { rowkey_indexer_.load(in); }

  void dump_rowkeys(const std::string& filename) const {
    rowkey_indexer_.dump(filename);
  }

 private:
  cppml::MinCountStringIndexer rowkey_indexer_;
  int64_t processed_lines_;
};

class FeatureManager {
 public:
  FeatureManager(int min_count = 0) : feature_pipline_(min_count) {}

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

  void dump_rowkeys(const std::string& filename) const {
    feature_pipline_.dump_rowkeys(filename);
  }

  TransformedFeature transform(const std::string& line) const {
    return feature_pipline_.transform(line);
  }

  TransformedFeature transform(const RawFeature& feature) const {
    return feature_pipline_.transform(feature);
  }

 private:
  FeaturePipline feature_pipline_;
};

}  // namespace fe

#endif
