#include <Eigen/Dense>
#include <iostream>

int main() {
    Eigen::MatrixXf m(2, 2);
    m << 1.0f, 2.0f,
    3.0f, 4.0f;
    std::cout << m << '\n';
}