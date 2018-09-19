
#include "tensorflow/core/framework/op.h"

namespace tensorflow {

REGISTER_OP("KrankInput")
    .Input("input: string")
    .Output("records: int32")
    .Output("target: int32")
    .Output("labels: int64")
    .SetIsStateful()
    .Attr("feature_manager_path: string = ''")
    .Doc(R"doc(
)doc");

}  // namespace tensorflow
