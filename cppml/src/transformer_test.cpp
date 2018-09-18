
#include <gtest/gtest.h>

#include "transformer.h"

TEST(MinMaxScalerTest, basic) {
  cppml::MinMaxScaler trans(0.0, 2.0);
  trans.feed(0.0);
  trans.feed(2.0);

  double v = trans.transform(1.8).as_double();
  std::cout << v << std::endl;
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
