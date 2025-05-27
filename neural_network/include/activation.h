#include <armadillo>

#ifndef ACTIVATION_H
#define ACTIVATION_H

class Activation_ReLU {
    public:
        arma::mat m_output;
        arma::mat m_inputs;

        // constructor for 

        void forward(const arma::mat inputs);

        arma::mat backward(const arma::mat& dvalues);
};

class Activation_ReLU_Leaky {
    public:
        arma::mat m_output;
        arma::mat m_inputs;

        double m_alpha = 0.01; // Leaky ReLU parameter

        void forward(const arma::mat inputs);

        arma::mat backward(const arma::mat& dvalues);
};

#endif