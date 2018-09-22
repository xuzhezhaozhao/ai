
package(
    default_visibility = [
        "//tensorflow_serving:internal",
    ],
    features = ["-layering_check"],
)

cc_library(
    name = "krank_predict_input_ops",
    visibility = ["//visibility:public"],
    srcs = glob(["*.cc", "fe/*.cpp", "fe/*.h", "cppml/*.h"]),
    copts = ["-std=c++11 -I."],
    deps = ["@org_tensorflow//tensorflow/core:framework_headers_lib",
            "@org_tensorflow//tensorflow/core/util/ctc",
            "@org_tensorflow//third_party/eigen3",
    ],
    alwayslink = 1,
)
