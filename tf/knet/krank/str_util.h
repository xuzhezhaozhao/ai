
#ifndef _KNET_KRANK_STR_UTIL_H_
#define _KNET_KRANK_STR_UTIL_H_

#include <vector>

#include "stringpiece.h"

namespace krank {

std::vector<StringPiece> Split(StringPiece text, char delim);

}  // namespace krank

#endif
