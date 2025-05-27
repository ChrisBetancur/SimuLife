#include <activation.h>

void Activation_ReLU::forward(const arma::mat inputs) {
    m_output = inputs;
    m_inputs = inputs;
    
    for (size_t i = 0; i < m_output.n_rows; ++i) {
        for (size_t j = 0; j < m_output.n_cols; ++j) {
            m_output(i, j) = std::max(0.0, m_output(i, j));
        }
    }
}

arma::mat Activation_ReLU::backward(const arma::mat& dvalues) {
    // Convert boolean mask to double values (1.0/0.0)
    arma::mat drelu = arma::conv_to<arma::mat>::from(m_inputs > 0.0);
    return dvalues % drelu;  // Element-wise multiplication
}

void Activation_ReLU_Leaky::forward(const arma::mat inputs) {
    m_inputs = inputs;
    m_output = inputs;

    m_output.transform([this](double val) { 
        return val > 0 ? val : m_alpha * val;
    });
}

arma::mat Activation_ReLU_Leaky::backward(const arma::mat& dvalues) {
    arma::mat drelu = arma::mat(m_inputs.n_rows, m_inputs.n_cols, arma::fill::ones);
    drelu.elem(arma::find(m_inputs <= 0)).fill(m_alpha);  // Vectorized operation
    return dvalues % drelu;  // Element-wise multiplication
}