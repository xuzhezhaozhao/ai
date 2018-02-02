
#include "tensorflow/core/framework/op.h"

namespace tensorflow {

REGISTER_OP("Fasttext")
    .Output("vocab_word: string")
    .Output("vocab_freq: int32")
    .Output("words_per_epoch: int64")
    .Output("current_epoch: int32")
    .Output("total_tokens_processed: int64")
    .Output("examples: int32")
    .Output("labels: int32")
    .Output("valid_lengths: int32")
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
    .Attr("seed: int = 1")
    .Doc(R"doc(
Fasttext custome operator.
)doc");


REGISTER_OP("NegTrainWord2vec")
    .Input("w_in: Ref(float)")
    .Input("w_out: Ref(float)")
    .Input("examples: int32")
    .Input("labels: int32")
    .Input("lr: float")
    .Output("loss: float")
    .SetIsStateful()
    .Attr("vocab_count: list(int)")
    .Attr("num_negative_samples: int")
    .Doc(R"doc(
Training via negative sampling.

w_in: input word embedding.
w_out: output word embedding.
examples: A vector of word ids.
labels: A vector of word ids.
vocab_count: Count of words in the vocabulary.
num_negative_samples: Number of negative samples per example.
)doc");

}  // namespace tensorflow
