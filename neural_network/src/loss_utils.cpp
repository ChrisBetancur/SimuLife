#include <loss_utils.h>

double mse_loss(const arma::mat& acc, const arma::mat& pred) {
    if (acc.n_rows != pred.n_rows || acc.n_cols != pred.n_cols) {
        std::cerr << "Error: Input size does not match weights size" << std::endl;
        return std::numeric_limits<double>::quiet_NaN();
    }

    arma::mat diff = acc - pred;
    double loss = arma::accu(diff % diff) / diff.n_elem;
    return loss;
}

arma::mat derivative_mse_loss(const arma::mat& acc, const arma::mat& pred) {
    if (acc.n_rows != pred.n_rows || acc.n_cols != pred.n_cols) {
        throw std::invalid_argument("Input matrices must have same dimensions");
    }
    
    arma::mat diff = acc - pred;
    return (2.0 * diff) / diff.n_elem; // (∂L/∂output) for MSE
}

double regularization_loss(const LayerDense& layer) {
    double loss = 0.0;

    if (layer.m_weight_regularizer_L1 > 0) {
        loss += layer.m_weight_regularizer_L1 * arma::accu(arma::abs(layer.m_weights));
    }
    if (layer.m_weight_regularizer_L2 > 0) {
        loss += layer.m_weight_regularizer_L2 * arma::accu(layer.m_weights % layer.m_weights);
    }
    if (layer.m_bias_regularizer_L1 > 0) {
        loss += layer.m_bias_regularizer_L1 * arma::accu(arma::abs(layer.m_biases));
    }
    if (layer.m_bias_regularizer_L2 > 0) {
        loss += layer.m_bias_regularizer_L2 * arma::accu(layer.m_biases % layer.m_biases);
    }

    return loss;
}