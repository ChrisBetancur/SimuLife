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
/*int main() {
    arma::mat inputs = {
        { 0.1, -0.2,  0.3, -0.4,  0.5},
        {-0.6, -0.7, -0.8, -0.9, -1.0},
        { 1.1,  1.2,  1.3,  1.4,  1.5}   
    };

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
}*/

int main() {
    // --- dummy batch of 3 examples with 7 inputs each
    arma::mat inputs = {
        { 0.1, -0.2,  0.3, -0.4,  0.5, -0.6,  0.7},
        {-0.6, -0.7, -0.8, -0.9, -1.0,  1.1, -1.2},
        { 1.1,  1.2,  1.3,  1.4,  1.5, -1.6,  1.7}
    };

    // --- pretend “true Q-values” for these 3 states and 4 actions
    arma::mat expected_output = {
        { 1.0, 0.5, -0.2, 0.0},
        { 0.2, 1.2,  0.0, 0.1},
        {-0.1,0.3,  0.7, 0.9}
    };

    inputs.print("\nInputs (3×7):");

    // --- hyperparams
    const int    num_epochs    = 200000;
    const double lr_initial    = 1e-3;
    // l2 reg is used to prevent overfitting, small penalty
    // for small data set, strong regularization is not needed
    // but we still use it to prevent overfitting
    // if you use a larger data set, you can increase the l2 reg
    // to 0.0001 or 0.001
    const double l2_reg        = 5e-5; // with tiny data set, strong regularilizaiton is not needed
    const double beta1         = 0.9, beta2 = 0.999;
    const double eps           = 1e-8;

    arma::arma_rng::set_seed_random();
    /*arma::mat weights1(7, 64, arma::fill::value(0.1));  // All weights = 0.1
    arma::mat biases1(1, 64, arma::fill::value(0.1));   // All biases = 0.1
    arma::mat weights2(64, 32, arma::fill::value(0.1));  // All weights = 0.1
    arma::mat biases2(1, 32, arma::fill::value(0.1));    // All biases = 0.1
    arma::mat weights3(32, 4, arma::fill::value(0.1));  // All weights = 0.1
    arma::mat biases3(1, 4, arma::fill::value(0.1));*/

    arma::mat weights1 = arma::randn<arma::mat>(7,64) * std::sqrt(2.0/7);
    arma::mat biases1(1, 64, arma::fill::value(0.1));
    arma::mat weights2 = arma::randn<arma::mat>(64,32) * std::sqrt(2.0/64);
    arma::mat biases2(1, 32, arma::fill::value(0.1));
    arma::mat weights3 = arma::randn<arma::mat>(32,4) * std::sqrt(2.0/32);
    arma::mat biases3(1, 4, arma::fill::value(0.1));

    // --- Layer sizes: 7 → 64 → 32 → 4
    LayerDense layer1(7,  64,  0.0, l2_reg, 0.0, 0);
    layer1.set_weights(weights1);
    layer1.set_biases(biases1);
    Activation_ReLU_Leaky act1;

    LayerDense layer2(64, 32,  0.0, l2_reg, 0.0, 0);
    layer2.set_weights(weights2);
    layer2.set_biases(biases2);
    Activation_ReLU_Leaky act2;

    LayerDense layer3(32, 4,   0.0, l2_reg, 0.0, 0);
    layer3.set_weights(weights3);
    layer3.set_biases(biases3);
    // final head is linear (no activation)

    // --- optimizer: Adam
    Optimizer_Adam optimizer(lr_initial, beta1, beta2, eps);

    // --- training loop
    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        // forward
        layer1.forward(inputs);
        // print output after first layer if NaN
        if (layer1.m_output.has_nan()) {
            // print epoch
            std::cerr << "Error: Layer 1 output contains NaN values at epoch " << epoch << "." << std::endl;
            std::cerr << "Error: Layer 1 output contains NaN values." << std::endl;
            std::cerr << "Layer 1 output:\n";
            layer1.m_output.print();
            // print weights and biases if NaN
            if (layer1.m_weights.has_nan() || layer1.m_biases.has_nan())
            {
                std::cerr << "Error: Layer 1 weights or biases contain NaN values." << std::endl;
                std::cerr << "Layer 1 weights:\n";
                layer1.m_weights.print();
                std::cerr << "Layer 1 biases:\n";
                layer1.m_biases.print();
            }
            exit(1);
        }
        act1.forward(layer1.m_output);

        // print output after first layer if NaN
        if (act1.m_output.has_nan()) {
            std::cerr << "Error: Activation output after first layer contains NaN values." << std::endl;
            std::cerr << "Activation output:\n";
            act1.m_output.print();
            exit(1);
        }

        layer2.forward(act1.m_output);
        act2.forward(layer2.m_output);

        layer3.forward(act2.m_output);
        arma::mat &predictions = layer3.m_output;

        // loss = MSE + L2 regs
        double loss = mse_loss(predictions, expected_output)
                    + regularization_loss(layer1)
                    + regularization_loss(layer2)
                    + regularization_loss(layer3);


        // print predictions if they are NaN
        if (predictions.has_nan()) {
            std::cerr << "Error: Predictions contain NaN values." << std::endl;
            std::cerr << "Predictions:\n";
            predictions.print();
            exit(1);
        }
        // backward
        arma::mat d_loss  = derivative_mse_loss(predictions, expected_output);
        layer3.backward(d_loss);

        arma::mat d2 = act2.backward(layer3.m_dinputs);
        layer2.backward(d2);

        arma::mat d1 = act1.backward(layer2.m_dinputs);
        layer1.backward(d1);

        optimizer.pre_update_params();



        // update
        optimizer.update(layer1);
        optimizer.update(layer2);
        optimizer.update(layer3);



        // every 50k print
        if ((epoch + 1) % 50000 == 0) {
            std::cout << "Epoch " << (epoch+1)
                      << " loss: " << loss << std::endl;
        }
    }

    // final output
    std::cout << "\nFinal predictions:\n";
    /*layer1.forward(inputs);
    act1.forward(layer1.m_output);
    layer2.forward(act1.m_output);
    act2.forward(layer2.m_output);
    layer3.forward(act2.m_output);*/
    layer3.m_output.print();

    return 0;
}
