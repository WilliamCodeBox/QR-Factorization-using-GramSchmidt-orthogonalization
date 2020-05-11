#if !defined(CPP_PROJECT_TEMPLATE_TEST_DEMO_H)
#define CPP_PROJECT_TEMPLATE_TEST_DEMO_H

#include <gtest/gtest.h>

#include "gram-schmidt.hpp"

using namespace arma;

TEST(QRFactorization, gram_schmidt_orthogonalization_1) {
    vec a({1, 1, 1});
    vec b({1, 0, 2});

    mat A(3, 2);
    A.col(0) = a;
    A.col(1) = b;

    mat orthA(3, 2, arma::fill::zeros);
    GramSchmidt(A, orthA);

    std::cout << orthA(0, 0) << std::endl;

    double eps = 1.0e-4;

    ASSERT_NEAR(orthA(0, 0), 5.7735e-01, eps);
    ASSERT_NEAR(orthA(1, 0), 5.7735e-01, eps);
    ASSERT_NEAR(orthA(2, 0), 5.7735e-01, eps);

    ASSERT_NEAR(orthA(0, 1), 0.0, eps);
    ASSERT_NEAR(orthA(1, 1), -7.0711e-01, eps);
    ASSERT_NEAR(orthA(2, 1), 7.0711e-01, eps);
}

TEST(QRFactorization, gram_schmidt_orthogonalization_2) {
    vec a({1, 1, 1});
    vec b({0, 2, 0});
    vec c({1, 0, 3});

    mat A(3, 3, arma::fill::zeros);
    A.col(0) = a;
    A.col(1) = b;
    A.col(2) = c;

    mat orthoNormA(3, 3, arma::fill::zeros);
    GramSchmidt(A, orthoNormA);

    double eps = 1.0e-4;
    ASSERT_NEAR(orthoNormA(0, 0), 5.7735e-01, eps);
    ASSERT_NEAR(orthoNormA(1, 0), 5.7735e-01, eps);
    ASSERT_NEAR(orthoNormA(2, 0), 5.7735e-01, eps);

    ASSERT_NEAR(orthoNormA(0, 1), -4.0825e-01, eps);
    ASSERT_NEAR(orthoNormA(1, 1), 8.1650e-01, eps);
    ASSERT_NEAR(orthoNormA(2, 1), -4.0825e-01, eps);

    ASSERT_NEAR(orthoNormA(0, 2), -7.0711e-01, eps);
    ASSERT_NEAR(orthoNormA(1, 2), 0.0, eps);
    ASSERT_NEAR(orthoNormA(2, 2), 7.0711e-01, eps);
}

TEST(QRFactorization, gram_schmidt_orthogonalization_3) {
    mat B({{1, 2}, {1, 1}, {0, 1}});

    mat orthNormB(arma::size(B), arma::fill::zeros);
    GramSchmidt(B, orthNormB);

    double eps = 1.0e-4;

    ASSERT_NEAR(orthNormB(0, 0), 0.7071, eps);
    ASSERT_NEAR(orthNormB(1, 0), 0.7071, eps);
    ASSERT_NEAR(orthNormB(2, 0), 0.0, eps);

    ASSERT_NEAR(orthNormB(0, 1), 0.4082, eps);
    ASSERT_NEAR(orthNormB(1, 1), -0.4082, eps);
    ASSERT_NEAR(orthNormB(2, 1), 0.8165, eps);
}

#endif  // CPP_PROJECT_TEMPLATE_TEST_DEMO_H
