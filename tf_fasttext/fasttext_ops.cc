
#include "tensorflow/core/framework/op.h"

namespace tensorflow {

REGISTER_OP("Fasttext")
    .Output("vocab_word: string")
    .Output("vocab_freq: int32")
    .Output("words_per_epoch: int64")
    .Output("current_epoch: int32")
    .Output("total_words_processed: int64")
    .Output("examples: Texamples")
    .Output("labels: int32")
    .SetIsStateful()
    .Attr("train_data: string")
    .Attr("lr: float = 0.05")
    .Attr("lr_update_rate: int = 100")
    .Attr("dim: int = 100")
    .Attr("maxn: int = 0")
    .Attr("minn: int = 0")
    .Attr("word_ngrams: int = 1")
    .Attr("bucket: int = 2000000")
    .Attr("ws: int = 5")
    .Attr("epoch: int = 5")
    .Attr("min_count: int = 1")
    .Attr("neg: int = 5")
    .Attr("t: float = 1e-4")
    .Attr("verbose: int = 1")
    .Attr("min_count_label: int = 1")
    .Attr("label: string = '__label__'")
    .Attr("batch_size: int = 1")
    .Attr("Texamples: list({int32})")
    .Doc(R"doc(
Fasttext custome operator.
)doc");

}  // namespace tensorflow
