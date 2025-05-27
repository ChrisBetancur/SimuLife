#include <armadillo>

#ifndef LAYER_DENSE_H
#define LAYER_DENSE_H

class LayerDense {
    private:
        uint32_t n_inputs;
        uint32_t n_neurons;

    public:
        arma::mat m_output;
        arma::mat m_inputs; // Stores a_k^{l-1} (activations from previous layer)

        arma::mat m_dweights, m_dbiases, m_dinputs; // Gradients stored here

        arma::mat m_weights;
        arma::mat m_biases;

        // set velocity
        arma::mat m_velocity_weights;
        arma::mat m_velocity_biases;

        // lambda hyperparams for regularization
        double m_weight_regularizer_L1, m_weight_regularizer_L2;
        double m_bias_regularizer_L1, m_bias_regularizer_L2;

        LayerDense(uint32_t n_inputs, uint32_t n_neurons, double weight_regularizer_L1 = 0.0, double weight_regularizer_L2 = 0.0, 
            double bias_regularizer_L1 = 0.0, double bias_regularizer_L2 = 0.0);

        // Copy constructor
        LayerDense(const LayerDense&) = default;

        void set_weights(arma::mat weights);

        void set_biases(arma::mat biases);

        void forward(arma::mat inputs);

        void backward(arma::mat dvalues);
};

#endif // LAYER_DENSE_H