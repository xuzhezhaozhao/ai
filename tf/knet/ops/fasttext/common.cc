#include "common.h"

#include <fstream>

#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/random/distribution_sampler.h"
#include "tensorflow/core/lib/random/philox_random.h"
#include "tensorflow/core/lib/random/simple_philox.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/default/integral_types.h"
#include "tensorflow/core/platform/posix/posix_file_system.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/util/guarded_philox_random.h"

void PreProcessTrainData(const std::string& train_data_path,
                         std::shared_ptr<::fasttext::Dictionary> dict) {
  LOG(INFO) << "Preprocess train data beginning ...";
  std::ifstream ifs(train_data_path);
  if (!ifs.is_open()) {
    LOG(FATAL) << "Open " << train_data_path << " failed.";
  }
  dict->readFromFile(ifs);
  ifs.close();
  LOG(INFO) << "Preprocess train data done.";
}

void SaveDictionary(const std::string& dict_dir,
                    std::shared_ptr<::fasttext::Dictionary> dict) {
  auto root_dir = dict_dir;

  auto file_system = ::tensorflow::PosixFileSystem();
  if (file_system.FileExists(root_dir) != ::tensorflow::Status::OK()) {
    auto status = file_system.CreateDir(root_dir);
    if (status != ::tensorflow::Status::OK()) {
      LOG(FATAL) << "Create dir " << root_dir << " failed.";
    }
  }
  LOG(INFO) << "SaveDictionary to " << root_dir << " ...";
  auto saved_dict = ::tensorflow::io::JoinPath(root_dir, SAVED_DICT);
  auto dict_meta = ::tensorflow::io::JoinPath(root_dir, DICT_META);
  auto dict_words = ::tensorflow::io::JoinPath(root_dir, DICT_WORDS);
  auto dict_word_counts =
      ::tensorflow::io::JoinPath(root_dir, DICT_WORD_COUNTS);

  {
    // save dictionary
    LOG(INFO) << "Save dictionary to " << saved_dict << " ...";
    std::ofstream ofs(saved_dict);
    if (!ofs.is_open()) {
      LOG(FATAL) << "Open " << saved_dict << " failed.";
    }
    dict->save(ofs);
    if (!ofs.good()) {
      LOG(FATAL) << "write file " << saved_dict << " failed.";
    }
    ofs.close();
    LOG(INFO) << "Save dictionary OK";
  }

  {
    // save dict meta
    std::ofstream ofs(dict_meta);
    LOG(INFO) << "Write dict meta to " << dict_meta << " ...";
    if (!ofs.is_open()) {
      LOG(FATAL) << "Open " << dict_meta << " failed.";
    }

    int nwords = dict->nwords();
    int nlabels = dict->nlabels();
    int ntokens = dict->ntokens();
    int nvalidTokens = dict->nvalidTokens();
    auto to_write = std::string("nwords\t") + std::to_string(nwords) + "\n";
    ofs.write(to_write.data(), to_write.size());

    to_write = std::string("nlabels\t" + std::to_string(nlabels) + "\n");
    ofs.write(to_write.data(), to_write.size());

    to_write = std::string("ntokens\t" + std::to_string(ntokens) + "\n");
    ofs.write(to_write.data(), to_write.size());

    to_write =
        std::string("nvalidTokens\t" + std::to_string(nvalidTokens) + "\n");
    ofs.write(to_write.data(), to_write.size());

    if (!ofs.good()) {
      LOG(FATAL) << "write file " << dict_meta << " failed.";
    }

    ofs.close();
    LOG(INFO) << "Write dict meta OK";
  }

  {
    // save dict words
    std::ofstream ofs(dict_words);
    LOG(INFO) << "Write dict words to " << dict_words << " ...";
    if (!ofs.is_open()) {
      LOG(FATAL) << "Open " << dict_words << " failed.";
    }
    for (const auto& entry : dict->words()) {
      if (entry.type == ::fasttext::entry_type::word) {
        ofs.write(entry.word.data(), entry.word.size());
        ofs.write("\n", 1);
      }
    }
    if (!ofs.good()) {
      LOG(FATAL) << "write file " << dict_words << " failed.";
    }
    ofs.close();
    LOG(INFO) << "Write dict words OK";
  }
  {
    // save dict word counts
    std::ofstream ofs(dict_word_counts);
    LOG(INFO) << "Write dict word counts to " << dict_word_counts << " ...";
    if (!ofs.is_open()) {
      LOG(FATAL) << "Open " << dict_word_counts << " failed.";
    }
    for (const auto& entry : dict->words()) {
      if (entry.type == ::fasttext::entry_type::word) {
        auto s = std::to_string(entry.count);
        ofs.write(s.data(), s.size());
        ofs.write("\n", 1);
      }
    }
    if (!ofs.good()) {
      LOG(FATAL) << "write file " << dict_word_counts << " failed.";
    }
    ofs.close();
    LOG(INFO) << "Write dict words OK";
  }
}
