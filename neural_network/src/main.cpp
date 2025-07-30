#include <iostream>
#include <cstdint>
#include <armadillo>
#include <limits>
#include <cmath>
#include <layer_dense.h>
#include <activation.h>
#include <optimizer.h>
#include <loss_utils.h>


// Entry point of the program
int main() {
    arma::mat inputs = {
        { 0.1, -0.2,  0.3, -0.4,  0.5},
        {-0.6, -0.7, -0.8, -0.9, -1.0},
        { 1.1,  1.2,  1.3,  1.4,  1.5}   
    };

    // First output before optimizations
    /* Output before optimizations
        0.3600   0.3600
        0.1000   0.1000
        1.6000   1.6000
    */
    inputs.print("\nInputs (3x5):");

    arma::mat expected_output = {
        { 0.8, 0.9},
        { 0.1,  0.9},
        { 0.8,  0.9} 
    };

    // --------------------------------------------------
    // Layer 1: 5 inputs -> 20 hidden (constant init)
    // --------------------------------------------------
    arma::mat weights1(5, 20, arma::fill::value(0.1));  // All weights = 0.1
    arma::mat biases1(1, 20, arma::fill::value(0.1));   // All biases = 0.1

    arma::mat weights2(20, 10, arma::fill::value(0.1));  // All weights = 0.1
    arma::mat biases2(1, 10, arma::fill::value(0.1));    // All biases = 0.1

    arma::mat weights3(10, 2, arma::fill::value(0.1));  // All weights = 0.1
    arma::mat biases3(1, 2, arma::fill::value(0.1));    // All biases = 0.1

    

    // Input -> Hidden
    LayerDense layer1(5, 20, 0.0, 0.0001, 0.0, 0);
    layer1.set_weights(weights1);
    layer1.set_biases(biases1);
    //layer1.forward(inputs);
    //layer1.m_output.print("Hidden Layer Output");

    // Hidden -> Activation
    Activation_ReLU activation1;
    //activation1.forward(layer1.m_output);
    //activation1.m_output.print("After ReLU");

    // Hidden -> Output
    LayerDense layer2(20, 10, 0.0, 0.0001, 0.0, 0);
    layer2.set_weights(weights2);
    layer2.set_biases(biases2);
    //layer2.forward(activation1.m_output);
    //layer2.m_output.print("Final Output");


    // Output -> Activation
    Activation_ReLU activation2;
    //activation2.forward(layer2.m_output);
    //activation2.m_output.print("After ReLU");

    LayerDense layer3(10, 2, 0.0, 0.0001, 0.0, 0);
    layer3.set_weights(weights3);
    layer3.set_biases(biases3);

    Activation_ReLU activation3;

    int num_epochs = 500000;
    double starting_learning_rate = 0.1;
    // Learning rate decay
    double decay = 0.0001;
    double learning_rate = starting_learning_rate;

    int epoch = 0;

    Optimizer_SGD optimizer(learning_rate, decay, epoch, 0.3);

    // Training loop
    for (epoch = 0; epoch < num_epochs; ++epoch) {
        //double learning_rate = starting_learning_rate * (1.0 / (1.0 + decay * epoch));
        // Forward pass
        layer1.forward(inputs);
        activation1.forward(layer1.m_output);
        layer2.forward(activation1.m_output);
        activation2.forward(layer2.m_output);
        layer3.forward(activation2.m_output);
        activation3.forward(layer3.m_output);

        // Loss calculation
        double loss = mse_loss(activation3.m_output, expected_output);
        loss += regularization_loss(layer1);
        loss += regularization_loss(layer2);
        loss += regularization_loss(layer3);
        // print output before optimizations
        //layer2.m_output.print("Output before optimizations");

        // Backward pass
        arma::mat d_loss = derivative_mse_loss(activation3.m_output, expected_output);
        arma::mat d_act = activation3.backward(d_loss);

        layer3.backward(d_act);
        d_act = activation2.backward(layer3.m_dinputs);
        layer2.backward(d_act);
        d_act = activation1.backward(layer2.m_dinputs);
        layer1.backward(d_act);

        optimizer.pre_update_params();

        // Update parameters
        optimizer.update(layer1);
        optimizer.update(layer2);
        optimizer.update(layer3);

        optimizer.post_update_params();

        // Print loss every 100 epochs
        if (epoch == num_epochs - 1) {
            std::cout << "Epoch " << epoch << ", Loss: " << loss << std::endl;
        }
        
    }
    layer3.m_output.print("Output");


    return 0;
}
