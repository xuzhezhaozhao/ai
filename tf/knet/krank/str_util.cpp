
#include <string>

#include "str_util.h"

namespace krank {

std::vector<StringPiece> Split(StringPiece text, char delim) {
  std::vector<StringPiece> result;
  size_t token_start = 0;
  if (!text.empty()) {
    for (size_t i = 0; i < text.size() + 1; i++) {
      if ((i == text.size()) || (delim == text[i])) {
        StringPiece token(text.data() + token_start, i - token_start);
        result.push_back(token);
        token_start = i + 1;
      }
    }
  }
  return result;
}

}  // namespace krank
