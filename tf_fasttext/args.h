/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#ifndef FASTTEXT_ARGS_H
#define FASTTEXT_ARGS_H

#include <istream>
#include <ostream>
#include <string>
#include <vector>

namespace fasttext {

struct Args {
    std::string train_data;
    float lr = 0.05f;
    int lr_update_rate = 100;
    int dim = 100;
    int maxn = 0;
    int minn = 0;
    int word_ngrams = 1;
    int bucket = 2000000;
    int ws = 5;
    int epoch = 5;
    int min_count = 5;
    int neg = 5;
    float t = 1e-4f;
    int verbose = 1;
    int min_count_label = 5;
    std::string label = "__label__";

    int batch_size = 1;
};

}

#endif
