
#include <iostream>
#include <memory>
#include <string>

#include <grpc++/grpc++.h>

#include "tensorflow_serving/apis/predict.pb.h"
#include "tensorflow_serving/apis/prediction_service.grpc.pb.h"

using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;
using tensorflow::serving::PredictRequest;
using tensorflow::serving::PredictResponse;
using tensorflow::serving::PredictionService;
using tensorflow::TensorProto;
using tensorflow::TensorShapeProto;
using tensorflow::DataType;

class PredictionClient {
 public:
  PredictionClient(std::shared_ptr<Channel> channel)
      : stub_(PredictionService::NewStub(channel)) {}

  void Predict(const std::string& serve_name) {
    PredictRequest request;
    auto* model_spec = request.mutable_model_spec();
    model_spec->set_name(serve_name);
    model_spec->set_signature_name("predicts");
    auto* inputs = request.mutable_inputs();
    TensorProto input_tensor;
    TensorShapeProto shape;
    auto* dim = shape.add_dim();
    dim->set_size(1);
    dim = shape.add_dim();
    dim->set_size(10);

    auto* input_shape = input_tensor.mutable_tensor_shape();
    (*input_shape) = shape;

    input_tensor.set_dtype(DataType::DT_INT64);
    for (int i = 0; i < 10; i++) {
      input_tensor.add_int64_val(i);
    }

    (*inputs)["watched"] = input_tensor;

    PredictResponse response;
    ClientContext context;

    Status status = stub_->Predict(&context, request, &response);
    if (status.ok()) {
      std::cout << "OK" << std::endl;
      std::cout << response.DebugString() << std::endl;
    } else {
      std::cout << status.error_code() << ": " << status.error_message()
                << std::endl;
    }
  }

 private:
  std::unique_ptr<PredictionService::Stub> stub_;
};

int main(int argc, char* argv[]) {
  PredictionClient client(
      grpc::CreateChannel(argv[1], grpc::InsecureChannelCredentials()));
  client.Predict(argv[2]);

  return 0;
}
