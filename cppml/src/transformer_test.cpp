
#include <gtest/gtest.h>
#include <iostream>
#include <fstream>

#include "transformer.h"

TEST(MinMaxScalerTest, basic) {
  cppml::MinMaxScaler trans(0, 1);
  trans.feed(0.0);
  trans.feed(1.2);
  trans.feed(0.2);
  trans.feed(0.8);
  trans.feed(2.0);
  ASSERT_DOUBLE_EQ(trans.getMin(), 0.0);
  ASSERT_DOUBLE_EQ(trans.getMax(), 2.0);
  ASSERT_DOUBLE_EQ(trans.transform(-2.8).as_numeric(), 0.0);
  ASSERT_DOUBLE_EQ(trans.transform(-0.5).as_numeric(), 0.0);
  ASSERT_DOUBLE_EQ(trans.transform(0.0).as_numeric(), 0.0);
  ASSERT_DOUBLE_EQ(trans.transform(0.5).as_numeric(), 0.25);
  ASSERT_DOUBLE_EQ(trans.transform(1.0).as_numeric(), 0.5);
  ASSERT_DOUBLE_EQ(trans.transform(1.8).as_numeric(), 0.9);
  ASSERT_DOUBLE_EQ(trans.transform(2.0).as_numeric(), 1.0);
  ASSERT_DOUBLE_EQ(trans.transform(2.8).as_numeric(), 1.0);
}

TEST(MinCountStringIndexer, minCount0) {
  cppml::MinCountStringIndexer strindexer0;
  ASSERT_EQ(strindexer0.getMinCount(), 0);
  std::vector<std::string> strs = {"a", "b", "c", "d", "e"};
  for (const auto& s : strs) {
    strindexer0.feed(s);
  }
  strindexer0.feed_end();
  std::set<int> indexes;
  strs.push_back("unknown");
  for (const auto& s : strs) {
    indexes.insert(strindexer0.transform(s).as_integer());
  }
  ASSERT_EQ(indexes.size(), 6);
  ASSERT_EQ(indexes.count(0), 1);
  ASSERT_EQ(indexes.count(1), 1);
  ASSERT_EQ(indexes.count(2), 1);
  ASSERT_EQ(indexes.count(3), 1);
  ASSERT_EQ(indexes.count(4), 1);
  ASSERT_EQ(indexes.count(5), 1);
}
TEST(MinCountStringIndexer, minCount2) {
  cppml::MinCountStringIndexer strindexer2(2);
  ASSERT_EQ(strindexer2.getMinCount(), 2);
  std::vector<std::string> strs = {"a", "a", "b", "c", "d", "e"};
  for (const auto& s : strs) {
    strindexer2.feed(s);
  }
  strindexer2.feed_end();
  std::set<int> indexes;
  for (const auto& s : strs) {
    indexes.insert(strindexer2.transform(s).as_integer());
  }
  ASSERT_EQ(indexes.size(), 2);
  ASSERT_EQ(indexes.count(0), 1);
  ASSERT_EQ(indexes.count(1), 1);
}

TEST(MinCountStringIndexer, Serialize) {
  cppml::MinCountStringIndexer saver(2);
  ASSERT_EQ(saver.getMinCount(), 2);
  std::vector<std::string> strs = {"a", "a",  "b", "c", "c", "d", "e", "e"};
  for (const auto& s : strs) {
    saver.feed(s);
  }
  saver.feed_end();
  std::string test_file = "test_min_count_string_indexer_serialize.bin";
  std::ofstream ofs(test_file);
  saver.save(ofs);
  ofs.close();

  cppml::MinCountStringIndexer loader(0);
  ASSERT_EQ(loader.getMinCount(), 0);
  std::ifstream ifs(test_file, std::ios::in & std::ios::binary);
  loader.load(ifs);
  ASSERT_EQ(loader.getMinCount(), 2);
  std::set<int> indexes;
  for (const auto& s : strs) {
    indexes.insert(loader.transform(s).as_integer());
  }
  ASSERT_EQ(indexes.size(), 4);
  ASSERT_EQ(indexes.count(0), 1);
  ASSERT_EQ(indexes.count(1), 1);
  ASSERT_EQ(indexes.count(2), 1);
  ASSERT_EQ(indexes.count(3), 1);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
