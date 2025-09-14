#ifndef LAYER_DENSE_H
#define LAYER_DENSE_H

#include <sstream> // For std::ostringstream
#include <string>  // For std::string
#include <iostream> // For std::cout (for testing)
#include <armadillo> // For arma::mat, arma::vec etc.

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

        // For adam optimizer
        arma::mat m_weight_momentums;  // first moment (⎯v)
        arma::mat m_weight_cache;      // second moment (⎯s)
        arma::rowvec m_bias_momentums;
        arma::rowvec m_bias_cache;


        int gradient_counter = 0;

        // lambda hyperparams for regularization
        double m_weight_regularizer_L1, m_weight_regularizer_L2;
        double m_bias_regularizer_L1, m_bias_regularizer_L2;

        LayerDense(uint32_t n_inputs, uint32_t n_neurons, double weight_regularizer_L1 = 0.0, double weight_regularizer_L2 = 0.0, 
            double bias_regularizer_L1 = 0.0, double bias_regularizer_L2 = 0.0);

        // Copy constructor
        //LayerDense(const LayerDense&) = default;

        LayerDense(const LayerDense& other);


        void reset();

        void set_weights(arma::mat weights);

        void set_biases(arma::mat biases);

        void forward(arma::mat inputs);

        void backward(arma::mat dvalues);

        void set_dweights(arma::mat dweights) { m_dweights = dweights; }

        void set_dbiases(arma::mat dbiases) { m_dbiases = dbiases; }

        void set_dinputs(arma::mat dinputs) { m_dinputs = dinputs; }

        arma::mat get_dweights() const { return m_dweights; }

        arma::mat get_dbiases() const { return m_dbiases; }

        arma::mat get_dinputs() const { return m_dinputs; }
};

#endif // LAYER_DENSE_H