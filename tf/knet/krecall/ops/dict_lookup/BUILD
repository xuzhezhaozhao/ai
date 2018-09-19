
package(
    default_visibility = [
        "//tensorflow_serving:internal",
    ],
    features = ["-layering_check"],
)

cc_library(
    name = "dict_lookup_ops",
    visibility = ["//visibility:public"],
    srcs = glob(["*.cc"]),
    copts = ["-std=c++11"],
    deps = ["@org_tensorflow//tensorflow/core:framework_headers_lib",
            "@org_tensorflow//tensorflow/core/util/ctc",
            "@org_tensorflow//third_party/eigen3",
    ],
    alwayslink = 1,
)
