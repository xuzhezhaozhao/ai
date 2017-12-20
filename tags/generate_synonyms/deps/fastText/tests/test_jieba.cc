
#include "cppjieba/Jieba.hpp"

static const char *DICT_PATH = "../deps/cppjieba/dict/jieba.dict.utf8";
static const char *HMM_PATH = "../deps/cppjieba/dict/hmm_model.utf8";
static const char *USER_DICT_PATH = "../deps/cppjieba/dict/user.dict.utf8";
static const char *IDF_PATH = "../deps/cppjieba/dict/idf.utf8";
static const char *STOP_WORD_PATH = "../deps/cppjieba/dict/stop_words.utf8";

static std::string JiebaCut(const std::string &s) {
  cppjieba::Jieba jieba(DICT_PATH, HMM_PATH, USER_DICT_PATH, IDF_PATH,
                        STOP_WORD_PATH);
  std::string result;
  std::vector<std::string> words;
  jieba.Cut(s, words, true);
  for (auto &word : s) {
    result += word;
    result += ' ';
  }
  result += '\n';
  return result;
}

int main(int argc, char *argv[]) {
  std::string s = "我很好，会来的";
  auto result = JiebaCut(s);
  std::cout << result << std::endl;
  
  return 0;
}
