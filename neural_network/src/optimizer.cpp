#include <optimizer.h>
#include <layer_dense.h>


Optimizer_SGD::Optimizer_SGD(double learning_rate, double decay, double step, double momentum) : 
    m_learning_rate(learning_rate),
    m_decay(decay),
    m_step(step),
    m_momentum(momentum) {
}

void Optimizer_SGD::pre_update_params() {
    m_learning_rate = m_learning_rate  /(1.0 + m_decay * m_step);
}

void Optimizer_SGD::update(LayerDense& layer) {
    // If the gradient is positive, the loss increases as the weight increases. Subtracting reduces the weight to lower the loss.

    arma::mat weight_updates = m_momentum * layer.m_velocity_weights - m_learning_rate * layer.m_dweights.t();
    layer.m_velocity_weights = weight_updates;

    arma::mat bias_updates = m_momentum * layer.m_velocity_biases - m_learning_rate * layer.m_dbiases;
    layer.m_velocity_biases = bias_updates;

    layer.m_weights += weight_updates;
    layer.m_biases += bias_updates;
}


void Optimizer_SGD::post_update_params() {
    // Reset decay
    m_step += 1;
}