#include "qr-fact.hpp"

#include <cassert>

#include "gram-schmidt.hpp"

void QRFactorization(arma::mat &A, arma::mat &Q, arma::mat &R) {
    arma::uword n_cols = A.n_cols;
    arma::uword n_rows = A.n_rows;

    assert(n_rows >= n_cols);
    assert(Q.n_cols == n_cols && Q.n_rows == n_rows);
    assert(R.n_cols == n_cols && R.n_rows == n_cols);

    GramSchmidt(A, Q);

    for (arma::uword j = 0; j < n_cols; j++) {
        arma::vec v(n_rows);
        v = A.col(j);

        for (arma::uword i = 0; i < j; i++) {
            R(i, j) = arma::dot(Q.col(i), A.col(j));
            v = v - R(i, j) * Q.col(i);
        }

        R(j, j) = arma::norm(v, 2);
        Q.col(j) = v / R(j, j);
    }
}