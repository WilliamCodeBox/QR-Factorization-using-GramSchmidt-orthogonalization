#include <assert.h>

#include <gram-schmidt.hpp>

void GramSchmidt(arma::mat &A, arma::mat &orthonormalizedA) {
    assert(A.n_cols == orthonormalizedA.n_cols &&
           A.n_rows == orthonormalizedA.n_rows);

    int n_cols = A.n_cols;

    orthonormalizedA.col(0) = arma::normalise(A.col(0));

    for (int i = 1; i < n_cols; i++) {
        orthonormalizedA.col(i) = A.col(i);
        for (int j = i - 1; j > -1; j--) {
            double tempDot = dot(orthonormalizedA.col(j), A.col(i));
            orthonormalizedA.col(i) =
                orthonormalizedA.col(i) - orthonormalizedA.col(j) * tempDot;
        }

        orthonormalizedA.col(i) = arma::normalise(orthonormalizedA.col(i));
    }
}