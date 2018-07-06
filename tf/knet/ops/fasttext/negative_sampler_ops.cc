#include "tensorflow/core/framework/op.h"

namespace tensorflow {

REGISTER_OP("NegativeSampler")
    .Input("true_classes: int64")
    .Output("records: int32")
    .SetIsStateful()
    .Attr("counts: tensor")
    .Doc(R"doc(
NegativeSampler operator.
)doc");


}  // namespace tensorflow
