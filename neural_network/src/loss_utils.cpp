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

double huber_loss(const arma::mat& predictions, const arma::mat& targets, double delta) {
    // Calculate the element-wise difference
    arma::mat diff = predictions - targets;
    arma::mat abs_diff = arma::abs(diff);

    // Create a boolean mask for the two cases
    arma::umat small_error_mask = (abs_diff <= delta);
    arma::umat large_error_mask = (abs_diff > delta);

    // Calculate the loss for each case
    arma::mat small_loss = 0.5 * arma::square(diff);
    arma::mat large_loss = delta * (abs_diff - 0.5 * delta);

    // Combine the two parts using the masks
    arma::mat combined_loss = small_loss % small_error_mask + large_loss % large_error_mask;

    // Return the mean of the combined loss
    return arma::accu(combined_loss) / (predictions.n_elem);
}

arma::mat derivative_huber_loss(const arma::mat& predictions, const arma::mat& targets, double delta) {
    // Calculate the element-wise difference
    arma::mat diff = predictions - targets;
    arma::mat abs_diff = arma::abs(diff);

    // Create a boolean mask for the two cases
    arma::umat small_error_mask = (abs_diff <= delta);
    arma::umat large_error_mask = (abs_diff > delta);

    // Calculate the derivative for each case
    arma::mat small_deriv = diff;
    arma::mat large_deriv = delta * arma::sign(diff);

    // Combine the two parts using the masks
    arma::mat combined_deriv = small_deriv % small_error_mask + large_deriv % large_error_mask;

    return combined_deriv;
}