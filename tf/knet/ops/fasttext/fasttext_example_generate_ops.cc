#include "tensorflow/core/framework/op.h"

namespace tensorflow {

REGISTER_OP("FasttextExampleGenerate")
    .Input("input: string")
    .Output("records: int32")
    .Output("labels: int64")
    .Output("ntokens: float32")
    .SetIsStateful()
    .Attr("use_saved_dict: bool = true")
    .Attr("dict_dir: string")
    .Attr("train_data_path: string")
    .Attr("word_ngrams: int = 1")
    .Attr("ws: int = 5")
    .Attr("lower_ws: int = 1")
    .Attr("min_count: int = 1")
    .Attr("t: float = 1e-4")
    .Attr("verbose: int = 1")
    .Attr("min_count_label: int = 10")
    .Attr("label: string = '__label__'")
    .Attr("ntargets: int = 1")
    .Attr("sample_dropout: float = 0.5")
    .Doc(R"doc(
Fasttext example generate operator.
Args
  input: A Tensor of string.
  records: A Tensor of type int32 and shape [D, ws]
  labels: A Tensor of type int64 and shape [D, ntargets]
  ntokens: A Tensor of type float32 and shape [D, 1], used for SGD
  learning rate decay.
)doc");

}  // namespace tensorflow
