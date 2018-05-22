
#include "tensorflow/core/framework/op.h"

namespace tensorflow {

REGISTER_OP("FasttextExampleGenerate")
    .Input("input: string")
    .Output("records: int32")
    .Output("labels: int32")
    .SetIsStateful()
    .Attr("train_data_path: string")
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
Fasttext example generate operator.
)doc");

}  // namespace tensorflow
