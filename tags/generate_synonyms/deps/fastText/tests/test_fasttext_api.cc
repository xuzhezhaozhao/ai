
#include <iostream>

#include "../src/fasttext_api.h"

int main(int argc, char *argv[]) {
  (void)argc;

  fasttext::FastTextApi fapi;
  fapi.LoadModel(argv[1]);
  std::string test_data = "我喜欢姚明想看球赛";
  auto preditions = fapi.Predict(test_data, 10);

  for (auto it = preditions.cbegin(); it != preditions.cend(); ++it) {
    std::cout << it->second << " : " << it->first << std::endl;
  }

  return 0;
}
