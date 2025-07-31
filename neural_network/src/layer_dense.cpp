#include <layer_dense.h>

LayerDense::LayerDense(uint32_t n_inputs, uint32_t n_neurons, double weight_regularizer_L1, double weight_regularizer_L2, 
    double bias_regularizer_L1, double bias_regularizer_L2) :
    n_inputs(n_inputs), n_neurons(n_neurons),
    m_weight_regularizer_L1(weight_regularizer_L1),
    m_weight_regularizer_L2(weight_regularizer_L2),
    m_bias_regularizer_L1(bias_regularizer_L1),
    m_bias_regularizer_L2(bias_regularizer_L2) {    

    m_velocity_weights = arma::zeros<arma::mat>(n_inputs, n_neurons);
    m_velocity_biases = arma::zeros<arma::mat>(1, n_neurons);
}

void LayerDense::reset() {
    //m_inputs.reset();
    m_output.reset();
}

void LayerDense::set_weights(arma::mat weights) {
    m_weights = weights;
}

void LayerDense::set_biases(arma::mat biases) {
    m_biases = biases;
}

void LayerDense::forward(arma::mat inputs) {
    m_inputs = inputs;
    if (inputs.n_cols != m_weights.n_rows) {
        std::cerr << "Error: Input size does not match weights size" << std::endl;
        std::cerr << "Input cols: " << inputs.n_cols << std::endl;
        std::cerr << "Weights rows: " << m_weights.n_rows << std::endl;
        return;
    }
    m_output = inputs * m_weights;
    m_output.each_row() += m_biases;
}

void LayerDense::backward(arma::mat dvalues) {
    m_dweights = dvalues.t() * m_inputs; // Gradient w.r.t. weights
    m_dbiases = arma::sum(dvalues, 0);   // Gradient w.r.t. biases

    // Regularization gradients
    if (m_weight_regularizer_L1 > 0) {
        arma::mat temp = m_weight_regularizer_L1 * arma::sign(m_weights);
        m_dweights += temp.t();
    }
    if (m_weight_regularizer_L2 > 0) {
        arma::mat temp = 2.0 * m_weight_regularizer_L2 * m_weights;
        m_dweights += temp.t();
    }
    if (m_bias_regularizer_L1 > 0) {
        m_dbiases += m_bias_regularizer_L1 * arma::sign(m_biases);
    }
    if (m_bias_regularizer_L2 > 0) {
        m_dbiases += 2.0 * m_bias_regularizer_L2 * m_biases;
    }

    // Calculate gradient w.r.t. inputs
    m_dinputs = dvalues * m_weights.t(); // Total influence, sum all of these contributions for neurons m in layer l+1

    m_dweights = arma::clamp(m_dweights, -1.0, 1.0);
    m_dbiases = arma::clamp(m_dbiases, -1.0, 1.0);
}