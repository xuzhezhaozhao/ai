#include "tensorflow/core/framework/op.h"

namespace tensorflow {
REGISTER_OP("FasttextDictIdLookup")
    .Input("input: string")
    .Output("ids: int32")
    .SetIsStateful()
    .Attr("dict_dir: string")
    .Doc(R"doc(
Fasttext dict id lookup operator.
  )doc");
}  // namespace tensorflow
