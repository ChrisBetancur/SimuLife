#include <optimizer.h>
#include <layer_dense.h>


/*Optimizer_SGD::Optimizer_SGD(double learning_rate, double step, double momentum) : 
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
}*/

/* NeuralNetwork(uint32_t input_dim, uint32_t output_dim, uint32_t hidden_dim, 
                    uint32_t num_m_layers, uint32_t batch_size, uint32_t nn_type, 
                    double initial_lr, double beta1, double beta2, 
                    double eps, int max_steps, double min_lr)*/

Optimizer_Adam::Optimizer_Adam(double learning_rate,
                               double beta1,
                               double beta2,
                               double eps,
                               double step,
                               int max_steps,
                               double min_lr)
    : m_lr_scheduler(learning_rate, min_lr, max_steps),
      m_step(step),
      m_beta1(beta1),
      m_beta2(beta2),
      m_eps(eps) {
}

void Optimizer_Adam::pre_update_params() {
    m_step += 1;
    m_learning_rate = m_lr_scheduler.get_learning_rate(m_step);
}

void Optimizer_Adam::update(LayerDense &layer) {
    // If layer momentums not initialized, set to zero mats
    if (layer.m_weight_momentums.n_elem == 0) {
        layer.m_weight_momentums = arma::zeros<arma::mat>(layer.m_weights.n_rows, layer.m_weights.n_cols);
        layer.m_weight_cache     = arma::zeros<arma::mat>(layer.m_weights.n_rows, layer.m_weights.n_cols);
        layer.m_bias_momentums   = arma::zeros<arma::rowvec>(layer.m_biases.n_cols);
        layer.m_bias_cache       = arma::zeros<arma::rowvec>(layer.m_biases.n_cols);
    }

    // Update momentums with current gradients
    layer.m_weight_momentums = m_beta1 * layer.m_weight_momentums + (1.0 - m_beta1) * layer.m_dweights.t();
    layer.m_bias_momentums   = m_beta1 * layer.m_bias_momentums   + (1.0 - m_beta1) * layer.m_dbiases;

    // Update cache (RMS prop)
    layer.m_weight_cache = m_beta2 * layer.m_weight_cache + (1.0 - m_beta2) * arma::square(layer.m_dweights.t());
    layer.m_bias_cache   = m_beta2 * layer.m_bias_cache   + (1.0 - m_beta2) * arma::square(layer.m_dbiases);


    // Correct bias in moment estimates
    arma::mat weight_momentums_corrected = layer.m_weight_momentums / (1.0 - std::pow(m_beta1, m_step));
    arma::rowvec bias_momentums_corrected = layer.m_bias_momentums   / (1.0 - std::pow(m_beta1, m_step));
    arma::mat weight_cache_corrected      = layer.m_weight_cache     / (1.0 - std::pow(m_beta2, m_step));
    arma::rowvec bias_cache_corrected     = layer.m_bias_cache       / (1.0 - std::pow(m_beta2, m_step));

    // Parameter update
    layer.m_weights -= m_learning_rate * weight_momentums_corrected /
                       (arma::sqrt(weight_cache_corrected) + m_eps);

    layer.m_biases  -= m_learning_rate * bias_momentums_corrected   /
                       (arma::sqrt(bias_cache_corrected)   + m_eps);


    // print check for all values that may cause gradient explosion (nan or inf)
    if (layer.m_weights.has_nan() || layer.m_weights.has_inf()) {
        std::cerr << "Error: Weights contain NaN or Inf values after Adam update." << std::endl;
        std::cerr << "Weights:\n";
        layer.m_weights.print();
        exit(1);
    }
    if (layer.m_biases.has_nan() || layer.m_biases.has_inf()) {
        std::cerr << "Error: Biases contain NaN or Inf values after Adam update." << std::endl;
        std::cerr << "Biases:\n";
        layer.m_biases.print();
        exit(1);
    }
    if (layer.m_weight_momentums.has_nan() || layer.m_weight_momentums.has_inf()) {
        std::cerr << "Error: Weight momentums contain NaN or Inf values after Adam update." << std::endl;
        std::cerr << "Weight momentums:\n";
        layer.m_weight_momentums.print();
        exit(1);
    }
    if (layer.m_bias_momentums.has_nan() || layer.m_bias_momentums.has_inf()) {
        std::cerr << "Error: Bias momentums contain NaN or Inf values after Adam update." << std::endl;
        std::cerr << "Bias momentums:\n";
        layer.m_bias_momentums.print();
        exit(1);
    }
    if (layer.m_weight_cache.has_nan() || layer.m_weight_cache.has_inf()) {
        std::cerr << "Error: Weight cache contain NaN or Inf values after Adam update." << std::endl;
        std::cerr << "Weight cache:\n";
        layer.m_weight_cache.print();
        exit(1);
    }
    if (layer.m_bias_cache.has_nan() || layer.m_bias_cache.has_inf()) {
        std::cerr << "Error: Bias cache contain NaN or Inf values after Adam update." << std::endl;
        std::cerr << "Bias cache:\n";
        layer.m_bias_cache.print();
        exit(1);
    }
    
}

void Optimizer_Adam::post_update_params() {
    //m_step += 1;
}