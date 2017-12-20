
#include <vector>
#include <string>

#include "mockmain.h"

int main(int argc, char** argv) {
  std::vector<std::string> args(argv, argv + argc);

  fasttext::mockmain(args);

  return 0;
}
