#ifndef LOSS_UTILS_H
#define LOSS_UTILS_H

#include <armadillo>
#include <layer_dense.h>

double mse_loss(const arma::mat& acc, const arma::mat& pred);

arma::mat derivative_mse_loss(const arma::mat& acc, const arma::mat& pred);

double regularization_loss(const LayerDense& layer);

#endif