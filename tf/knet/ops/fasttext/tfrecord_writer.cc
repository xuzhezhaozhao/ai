
#include "tensorflow/core/lib/io/record_writer.h"
#include "tensorflow/core/example/example.pb.h"
#include "tensorflow/core/platform/env.h"

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"

class WritableFile;

int main(int argc, char *argv[]) {

  // Create Example
  ::tensorflow::Example example;
  auto features = example.mutable_features();
  auto feature = features->mutable_feature();
  ::tensorflow::Feature words;
  auto bytes_list = words.mutable_bytes_list();
  int ws = 20;
  for (int i = 0; i < ws; ++i) {
    bytes_list->add_value("a");
  }
  for (int i = bytes_list->value_size(); i < ws; ++i) {
    bytes_list->add_value("");
  }
  (*feature)["words"] = words;

  std::string serialized;
  example.SerializeToString(&serialized);


  ::tensorflow::Env* env = ::tensorflow::Env::Default();
  std::string fname = "example.tfrecord";

  std::unique_ptr<::tensorflow::WritableFile> file;
  TF_CHECK_OK(env->NewWritableFile(fname, &file));
  ::tensorflow::io::RecordWriter writer(file.get());
  TF_CHECK_OK(writer.WriteRecord(serialized));
  TF_CHECK_OK(writer.Flush());

  return 0;
}
