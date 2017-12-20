
#include "fasttext_api.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#include "cppjieba/Jieba.hpp"

#include "fasttext.h"
#include "mockmain.h"

namespace fasttext {

static const char *DICT_PATH = "../dict/jieba.dict.utf8";
static const char *HMM_PATH = "../dict/hmm_model.utf8";
static const char *USER_DICT_PATH = "../dict/user.dict.utf8";
static const char *IDF_PATH = "../dict/idf.utf8";
static const char *STOP_WORD_PATH = "../dict/stop_words.utf8";

static cppjieba::Jieba *jieba = NULL;
static std::unordered_set<std::string> *stop_words = NULL;

static void LoadStopWords(const std::string stop_word_path,
                          std::unordered_set<std::string> &stop_words) {
  std::ifstream ifs(stop_word_path.c_str());
  if (!ifs.is_open()) {
    std::cerr << "stop word file: " << stop_word_path << " open faield."
              << std::endl;
    exit(EXIT_FAILURE);
  }
  std::string line;
  while (getline(ifs, line)) {
    stop_words.insert(line);
  }
}

static std::string JiebaCut(const std::string &sentence) {
  if (!jieba) {
    jieba = new cppjieba::Jieba(DICT_PATH, HMM_PATH, USER_DICT_PATH, IDF_PATH,
                                STOP_WORD_PATH);
    stop_words = new std::unordered_set<std::string>();
    LoadStopWords(STOP_WORD_PATH, *stop_words);
  }

  std::string result;
  std::vector<cppjieba::Word> words;
  jieba->Cut(sentence, words, true);
  for (auto &word : words) {
    if (stop_words->find(word.word) != stop_words->end()) {
      continue;
    }
    result += word.word;
    result += ' ';
  }
  result += '\n';
  return result;
}

void FastTextApi::predictString(
    const std::string &test, int k,
    std::vector<std::pair<real, std::string>> &predictions,
    bool usejieba) {
  if (!model_loaded_) {
    return;
  }

  std::string result = usejieba ? JiebaCut(test) : test;
  std::stringstream iss(result);
  fasttext_.predict(iss, k, predictions);
}

std::vector<real> FastTextApi::GetSentenceVector(const std::string &sentence) {
  if (!model_loaded_) {
    return std::vector<real>();
  }
  std::string result = JiebaCut(sentence);
  std::stringstream iss(result);
  Vector svec(fasttext_.getDimension());
  fasttext_.getSentenceVector(iss, svec);
  return std::vector<real>(svec.data_, svec.data_ + svec.m_);
}

FastTextApi::FastTextApi() {}

void FastTextApi::Supervised(const FastTextApi::SupervisedArgs &a) {
  std::vector<std::string> args;
  args.push_back("fasttext");
  args.push_back("supervised");
  args.push_back("-input");
  args.push_back(a.input);
  args.push_back("-output");
  args.push_back(a.output);
  args.push_back("-minCount");
  args.push_back(std::to_string(a.minCount));
  args.push_back("-minCountLabel");
  args.push_back(std::to_string(a.minCountLabel));
  args.push_back("-wordNgrams");
  args.push_back("-bucket");
  args.push_back(std::to_string(a.bucket));
  args.push_back("-minn");
  args.push_back(std::to_string(a.minn));
  args.push_back("-maxn");
  args.push_back(std::to_string(a.maxn));
  args.push_back("-t");
  args.push_back(std::to_string(a.t));
  args.push_back("-label");
  args.push_back(a.label);
  args.push_back("-lr");
  args.push_back(std::to_string(a.lr));
  args.push_back("-lrUpdateRate");
  args.push_back(std::to_string(a.lrUpdateRate));
  args.push_back("-dim");
  args.push_back(std::to_string(a.dim));
  args.push_back("-ws");
  args.push_back(std::to_string(a.ws));
  args.push_back("-epoch");
  args.push_back(std::to_string(a.epoch));
  args.push_back("-neg");
  args.push_back(std::to_string(a.neg));
  args.push_back("-loss");
  args.push_back(a.loss);
  args.push_back("-thread");
  args.push_back(std::to_string(a.thread));
  args.push_back("-cutoff");
  args.push_back(std::to_string(a.cutoff));
  args.push_back("-retrain");
  args.push_back(std::to_string(a.retrain));
  args.push_back("-qnorm");
  args.push_back(std::to_string(a.qnorm));
  args.push_back("-qout");
  args.push_back(std::to_string(a.qout));
  args.push_back("-dsub");
  args.push_back(std::to_string(a.dsub));

  mockmain(args);
}

void FastTextApi::Test(const FastTextApi::TestArgs &a) {
  std::vector<std::string> args;
  args.push_back("fasttext");
  args.push_back("test");
  args.push_back(a.model);
  args.push_back(a.test_data);
  args.push_back(std::to_string(a.k));

  mockmain(args);
}

std::vector<std::pair<real, std::string>> FastTextApi::Predict(
    const std::string &test, int k, bool usejieba) {
  std::vector<std::pair<real, std::string>> predictions;
  predictString(test, k, predictions, usejieba);
  for (auto it = predictions.begin(); it != predictions.end(); it++) {
    it->first = static_cast<float>(exp(it->first));
  }
  return predictions;
}

void FastTextApi::LoadModel(const std::string &model) {
  if (!model_loaded_) {
    fasttext_.loadModel(model);
    model_loaded_ = true;
  }
}

std::vector<std::pair<real, std::string>> FastTextApi::NN(
    const std::string &query, int k) {
  std::vector<std::pair<real, std::string>> nn;
  fasttext_.nnwithquery(query, k, 0, nn);
  return nn;
}

void FastTextApi::PrecomputeWordVectors() {
  fasttext_.precomputeWordVectors();
}

}  // namespace fasttext
