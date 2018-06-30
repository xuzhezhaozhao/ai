#include <fstream>

#include <gflags/gflags.h>
#include <spawn.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <thread>

#include "tensorflow/core/example/example.pb.h"
#include "tensorflow/core/lib/io/record_writer.h"
#include "tensorflow/core/platform/env.h"

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/posix/posix_file_system.h"

#include "args.h"
#include "common.h"
#include "defines.h"
#include "dictionary.h"

DEFINE_string(tfrecord_file, "", "");
DEFINE_int32(ws, 5, "");
DEFINE_int32(lower_ws, 1, "");
DEFINE_int32(min_count, 5, "");
DEFINE_double(t, 0.0001, "");
DEFINE_int32(ntargets, 1, "");
DEFINE_double(sample_dropout, 0.5, "");
DEFINE_string(train_data_path, "", "");
DEFINE_string(dict_dir, "", "");
DEFINE_int32(threads, 1, "");
DEFINE_int32(is_delete, 0, "");
DEFINE_int32(use_saved_dict, 0, "");

std::atomic<int64_t> line_processed;
std::atomic<int64_t> total;

class WritableFile;

static bool run_cmd(char *cmd) {
  pid_t pid;
  char sh[4] = "sh";
  char arg[4] = "-c";
  char *argv[] = {sh, arg, cmd, NULL};
  LOG(ERROR) << "Run command: " << cmd;
  int status = posix_spawn(&pid, "/bin/sh", NULL, NULL, argv, environ);
  if (status == 0) {
    LOG(ERROR) << "Child pid: " << pid;
    if (waitpid(pid, &status, 0) != -1) {
      LOG(ERROR) << "Child exited with status " << status;
    } else {
      LOG(ERROR) << "Child exited with status " << status
                 << ", errmsg = " << strerror(errno);
      return false;
    }
  } else {
    LOG(ERROR) << "posix_spawn failed, errmsg = " << strerror(status);
    return false;
  }
  return true;
}

void parse_and_save_dictionary(const std::string &train_data_path,
                               const std::string &dict_dir,
                               std::shared_ptr<::fasttext::Dictionary> dict) {
  LOG(INFO) << "Parse dictionary from " << train_data_path << " ...";
  PreProcessTrainData(train_data_path, dict);
  SaveDictionary(dict_dir, dict);
  LOG(INFO) << "Load dictionary OK.";
}

void load_dictionary(const std::string &dict_dir,
                     std::shared_ptr<::fasttext::Dictionary> dict) {
  // load dictionary
  auto root_dir = dict_dir;
  auto saved_dict = ::tensorflow::io::JoinPath(root_dir, SAVED_DICT);
  LOG(INFO) << "Load dictionary from " << saved_dict << " ...";
  std::ifstream ifs(saved_dict);
  if (!ifs.is_open()) {
    LOG(ERROR) << "Open saved_dict file '" << saved_dict << "' failed.";
    exit(-1);
  }
  dict->load(ifs);
  if (ifs.fail()) {
    LOG(ERROR) << "Load dictionary failed.";
    exit(-1);
  }
  ifs.close();
  LOG(INFO) << "Load dictionary OK";
}

inline int transform_id(int id) { return id + 1; }

void fill_example(const std::vector<int> &inst,
                  std::shared_ptr<::fasttext::Args> args,
                  ::tensorflow::Example *example) {
  auto features = example->mutable_features();
  auto feature = features->mutable_feature();
  ::tensorflow::Feature records;
  ::tensorflow::Feature label;
  for (int t = 0; t < args->ntargets; ++t) {
    int index = inst.size() - args->ntargets + t;
    label.mutable_int64_list()->add_value(transform_id(inst[index]));
  }
  (*feature)["label"] = label;

  // fill records
  auto records_int_list = records.mutable_int64_list();
  for (int i = 0; i < inst.size() - args->ntargets; ++i) {
    records_int_list->add_value(transform_id(inst[i]));
  }

  // padding records
  for (int i = inst.size() - args->ntargets; i < args->ws; ++i) {
    records_int_list->add_value(0);
  }
  (*feature)["records"] = records;
}

void dump_tfrecord(const std::string &input_file,
                   const std::string &tfrecord_file,
                   std::shared_ptr<::fasttext::Args> args,
                   std::shared_ptr<::fasttext::Dictionary> dict) {
  ::tensorflow::Env *env = ::tensorflow::Env::Default();
  std::unique_ptr<::tensorflow::WritableFile> file;
  TF_CHECK_OK(env->NewWritableFile(tfrecord_file, &file));
  ::tensorflow::io::RecordWriter writer(file.get());

  std::ifstream ifs(input_file);
  if (!ifs.is_open()) {
    LOG(FATAL) << "open " << input_file << " failed.";
  }
  std::string line;
  std::vector<int32_t> words;
  std::vector<int> inst;
  std::minstd_rand rng(time(NULL));
  while (!ifs.eof()) {
    ++line_processed;
    if (line_processed % 200000 == 0) {
      LOG(INFO) << line_processed << " processed," << total << " examples.";
    }

    std::getline(ifs, line);
    if (line == "") {
      continue;
    }
    words.clear();
    std::stringstream ss(line);
    int ntokens = dict->getLine(ss, words, rng);
    std::uniform_int_distribution<> uniform(args->lower_ws, args->ws);
    std::uniform_real_distribution<> dropout_uniform(0, 1);

    // genearte examples
    for (int w = 1; w < words.size(); w++) {
      inst.clear();
      if (dropout_uniform(rng) < args->sample_dropout) {
        continue;
      }
      // use words[w] as the first label
      int32_t boundary = std::min(w, uniform(rng));
      for (int c = -boundary; c < 0; c++) {
        inst.push_back(words[w + c]);
      }
      inst.push_back(words[w]);  // add label

      // TODO random select ntargets-1 labels
      for (int i = 0; i < args->ntargets - 1; ++i) {
        int t = w + 1 + i;
        if (t >= words.size()) {
          t = w;
        }
        inst.push_back(words[t]);
      }
      // Create Example and write to tfrecord file
      ::tensorflow::Example example;
      fill_example(inst, args, &example);
      std::string serialized;
      example.SerializeToString(&serialized);
      TF_CHECK_OK(writer.WriteRecord(serialized));
      ++total;
    }
  }

  // flush
  TF_CHECK_OK(writer.Flush());
  TF_CHECK_OK(writer.Close());
}

int main(int argc, char *argv[]) {
  google::ParseCommandLineFlags(&argc, &argv, true);

  line_processed.store(0);
  total.store(0);

  std::shared_ptr<::fasttext::Args> args = std::make_shared<::fasttext::Args>();
  args->ws = FLAGS_ws;
  args->lower_ws = FLAGS_lower_ws;
  args->min_count = FLAGS_min_count;
  args->t = FLAGS_t;
  args->ntargets = FLAGS_ntargets;
  args->sample_dropout = FLAGS_sample_dropout;
  args->dict_dir = FLAGS_dict_dir;

  std::shared_ptr<::fasttext::Dictionary> dict =
      std::make_shared<::fasttext::Dictionary>(args);
  if (FLAGS_use_saved_dict) {
    load_dictionary(args->dict_dir, dict);
  } else {
    parse_and_save_dictionary(FLAGS_train_data_path, args->dict_dir, dict);
  }

  if (FLAGS_threads == 1) {
    dump_tfrecord(FLAGS_train_data_path, FLAGS_tfrecord_file, args, dict);
  } else {
    const int kMaxCommandSize = 8192;
    static char cmd[kMaxCommandSize];

    std::string split_cmd =
        "split -a 3 -d -n l/" + std::to_string(FLAGS_threads) + " " +
        FLAGS_train_data_path + " " + FLAGS_train_data_path + ".";
    memcpy(cmd, split_cmd.data(), split_cmd.size());
    if (split_cmd.size() > kMaxCommandSize - 1) {
      LOG(FATAL) << "cmd buffer not enough.";
    }
    if (!run_cmd(cmd)) {
      LOG(FATAL) << "run split cmd failed.";
    }

    std::vector<std::thread> threads;
    char suffix[4];
    std::string todelete;
    for (int i = 0; i < FLAGS_threads; ++i) {
      snprintf(suffix, 4, "%03d", i);
      auto input_file = FLAGS_train_data_path + "." + suffix;
      auto tfrecord_file = FLAGS_tfrecord_file + "." + suffix;
      todelete += input_file + " ";
      threads.emplace_back(dump_tfrecord, input_file, tfrecord_file, args,
                           dict);
    }
    for (int i = 0; i < FLAGS_threads; ++i) {
      threads[i].join();
    }
    std::string rm_cmd = "rm -f " + todelete;
    memcpy(cmd, rm_cmd.data(), rm_cmd.size());
    if (FLAGS_is_delete) {
      LOG(ERROR) << "Delete splited files ...";
      if (!run_cmd(cmd)) {
        LOG(ERROR) << "failed delete files.";
      } else {
        LOG(ERROR) << "Delete splited files OK";
      }
    } else {
      LOG(ERROR) << "Don't delete splited files.";
    }
  }

  LOG(ERROR) << "dump " << total << " tfrecord examples.";

  return 0;
}
