#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {
REGISTER_OP("FasttextDictIdLookup")
    .Input("input: string")
    .Output("ids: int32")
    .SetIsStateful()
    .Attr("dict_dir: string")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* ctx) {
      ctx->set_output(0, ctx->input(0));
      return Status::OK();
    })
    .Doc(R"doc(
Fasttext dict id lookup operator.
  )doc");
}  // tensorflow
