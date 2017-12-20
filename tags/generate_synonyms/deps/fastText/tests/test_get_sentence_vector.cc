
#include <iostream>

#include "../src/fasttext_api.h"

int main(int argc, char *argv[]) {
  (void)argc;
  fasttext::FastTextApi fapi;
  fapi.LoadModel(argv[1]);
  std::string sentence = "我喜欢姚明想看球赛";

  // 获取文本向量
  auto svec = fapi.GetSentenceVector(sentence);
  std::cout << "size: " << svec.size() << std::endl;
  for (auto v : svec) {
    std::cout << v << " ";
  }
  std::cout << std::endl;

  return 0;
}
