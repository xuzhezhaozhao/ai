
#include "tensorflow/core/framework/op.h"

namespace tensorflow {

REGISTER_OP("StringSplitV2")
    .Input("str: string")
    .Output("splited: string")
    .Attr("delimter: string")
    .Doc(R"doc(
)doc");

}  // namespace tensorflow
