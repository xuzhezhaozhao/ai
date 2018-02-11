
#include <iostream>
#include "eigen/Eigen/Dense"
using Eigen::MatrixXd;
using Eigen::VectorXd;

int main()
{
  MatrixXd m(2,2);
  m(0,0) = 3;
  m(1,0) = 2.5;
  m(0,1) = -1;
  m(1,1) = m(1,0) + m(0,1);
  std::cout << m << std::endl;

  m = MatrixXd::Random(3, 3);
  m = (m + MatrixXd::Constant(3, 3, 1.2)) * 50;
  std::cout << "m = " << m << std::endl;
  VectorXd v(3);
  v << 1, 2, 3;
  std::cout << "m * v = " << m * v << std::endl;
}
