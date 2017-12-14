
#include <iostream>
#include <memory>
#include <string>

#include <grpc++/grpc++.h>

#include "tensorflow_serving/apis/prediction_service.grpc.pb.h"
#include "tensorflow_serving/apis/predict.pb.h"

using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;
using tensorflow::serving::PredictRequest;
using tensorflow::serving::PredictResponse;
using tensorflow::serving::PredictionService;

class PredictionClient {
  public:
    PredictionClient(std::shared_ptr<Channel> channel)
      : stub_(PredictionService::NewStub(channel)) {}

    void Predict() {
      PredictRequest request;

      PredictResponse response;
      ClientContext context;

      Status status = stub_->Predict(&context, request, &response);
      if (status.ok()) {
        std::cout << "OK" << std::endl;
      } else {
        std::cout << status.error_code() << ": " << status.error_message()
                  << std::endl;
      }
    }

  private:
    std::unique_ptr<PredictionService::Stub> stub_;
};

int main(int argc, char *argv[]) {
  PredictionClient client(grpc::CreateChannel(
        "localhost:9000", grpc::InsecureChannelCredentials()));
  client.Predict();

  return 0;
}
