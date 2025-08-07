#include <layer_dense.h>
#include <activation.h>
#include <optimizer.h>
#include <loss_utils.h>
#include <armadillo>
#include <vector>
#include <memory>
#include <io.h>

#define LEARNING_RATE 0.1
#define DECAY 0.0001
#define MOMENTUM 0.3

#define LR_INITIAL 1e-3
#define BETA1 0.9
#define BETA2 0.999
#define EPS 1e-8

class NeuralNetwork {

    public:
        std::vector<LayerDense> m_layers;
        std::vector<Activation_ReLU_Leaky> m_activations;
        //Optimizer_SGD optimizer;
        Optimizer_Adam optimizer;

        uint32_t m_nn_type; // 0 is online and 1 is target
        uint32_t m_batch_size;
        uint32_t m_input_dim;
        uint32_t num_layers;
        uint32_t m_output_dim;
        uint32_t m_hidden_dim;


        /*NeuralNetwork(uint32_t input_dim, uint32_t output_dim, uint32_t hidden_dim, 
                    uint32_t num_m_layers, uint32_t batch_size, uint32_t nn_type) :
            optimizer(LEARNING_RATE, DECAY, 0.0, MOMENTUM),
            m_nn_type(nn_type),
            m_batch_size(batch_size),
            m_input_dim(input_dim),
            num_layers(num_m_layers),
            m_output_dim(output_dim),
            m_hidden_dim(hidden_dim)
        {
            // Validate parameters
            if (num_m_layers < 3) {
                throw std::invalid_argument("Number of layers must be at least 3");
            }
            if (hidden_dim < 1) {
                throw std::invalid_argument("Hidden dimension must be at least 1");
            }

            // Reserve space upfront to avoid reallocations
            m_layers.reserve(num_m_layers);
            m_activations.reserve(num_m_layers);

            // Create input layer directly in vector
            m_layers.emplace_back(input_dim, hidden_dim, 0.0, 0.0001, 0.0, 0);
            auto& input_layer = m_layers.back();
            input_layer.set_weights(arma::mat(input_dim, hidden_dim, arma::fill::value(0.1)));
            input_layer.set_biases(arma::mat(1, hidden_dim, arma::fill::value(0.1)));

            // Create hidden layers
            for (uint32_t i = 0; i < num_m_layers - 2; ++i) {
                m_layers.emplace_back(hidden_dim, hidden_dim, 0.0, 0.0001, 0.0, 0);
                auto& layer = m_layers.back();
                layer.set_weights(arma::mat(hidden_dim, hidden_dim, arma::fill::value(0.1)));
                layer.set_biases(arma::mat(1, hidden_dim, arma::fill::value(0.1)));
            }

            // Create output layer
            m_layers.emplace_back(hidden_dim, output_dim, 0.0, 0.0001, 0.0, 0);
            auto& output_layer = m_layers.back();
            output_layer.set_weights(arma::mat(hidden_dim, output_dim, arma::fill::value(0.1)));
            output_layer.set_biases(arma::mat(1, output_dim, arma::fill::value(0.1)));

            // Create activations
            for (uint32_t i = 0; i < num_m_layers; ++i) {
                m_activations.emplace_back();
            }

            // Verify initialization
            for (const auto& layer : m_layers) {
                if (layer.m_weights.has_nan() || layer.m_biases.has_nan()) {
                    std::cerr << "Error: Layer weights or biases contain NaN values." << std::endl;
                    exit(1);
                }
            }
        }*/

        NeuralNetwork(uint32_t input_dim, uint32_t output_dim, uint32_t hidden_dim, 
                    uint32_t num_m_layers, uint32_t batch_size, uint32_t nn_type) :
            optimizer(LEARNING_RATE, DECAY, 0.0, MOMENTUM),
            m_nn_type(nn_type),
            m_batch_size(batch_size),
            m_input_dim(input_dim),
            num_layers(num_m_layers),
            m_output_dim(output_dim),
            m_hidden_dim(hidden_dim)
        {
            // Validate parameters
            if (num_m_layers < 3) {
                throw std::invalid_argument("Number of layers must be at least 3");
            }
            if (hidden_dim < 1) {
                throw std::invalid_argument("Hidden dimension must be at least 1");
            }

            // Reserve space upfront to avoid reallocations
            m_layers.reserve(num_m_layers);
            m_activations.reserve(num_m_layers);

            arma::arma_rng::set_seed_random();

            // Create input layer directly in vector
            m_layers.emplace_back(input_dim, hidden_dim, 0.0, 0.0001, 0.0, 0);
            auto& input_layer = m_layers.back();
            input_layer.set_weights(arma::randn<arma::mat>(input_dim,hidden_dim) * std::sqrt(2.0/7));
            input_layer.set_biases(arma::mat(1, hidden_dim, arma::fill::value(0.1)));

            // Create hidden layers
            for (uint32_t i = 0; i < num_m_layers - 2; ++i) {
                m_layers.emplace_back(hidden_dim, hidden_dim, 0.0, 5e-5, 0.0, 0);
                auto& layer = m_layers.back();
                layer.set_weights(arma::randn<arma::mat>(hidden_dim,hidden_dim) * std::sqrt(2.0/hidden_dim));
                layer.set_biases(arma::mat(1, hidden_dim, arma::fill::value(0.1)));
            }

            // Create output layer
            m_layers.emplace_back(hidden_dim, output_dim, 0.0, 5e-5, 0.0, 0);
            auto& output_layer = m_layers.back();
            output_layer.set_weights(arma::randn<arma::mat>(hidden_dim,output_dim) * std::sqrt(2.0/output_dim));
            output_layer.set_biases(arma::mat(1, output_dim, arma::fill::value(0.1)));

            // Create activations
            for (uint32_t i = 0; i < num_m_layers - 1; ++i) {
                m_activations.emplace_back();
            }

            // Verify initialization
            for (const auto& layer : m_layers) {
                if (layer.m_weights.has_nan() || layer.m_biases.has_nan()) {
                    std::cerr << "Error: Layer weights or biases contain NaN values." << std::endl;
                    exit(1);
                }
            }
        }

        NeuralNetwork(const NeuralNetwork& other) :
            optimizer(other.optimizer),
            m_nn_type(other.m_nn_type),
            m_batch_size(other.m_batch_size),
            m_input_dim(other.m_input_dim),
            num_layers(other.num_layers),
            m_output_dim(other.m_output_dim),
            m_hidden_dim(other.m_hidden_dim)
        {
            // Reserve space first
            m_layers.reserve(other.m_layers.size());
            m_activations.reserve(other.m_activations.size());
            
            // Copy layers using proper copy semantics
            for (const auto& layer : other.m_layers) {
                m_layers.push_back(layer);  // Uses fixed LayerDense copy constructor
            }
            
            // Copy activations
            for (const auto& activation : other.m_activations) {
                m_activations.push_back(activation);
            }
        }

        void reset_episode() {
            for (auto& layer : m_layers) {
                layer.m_inputs.reset();
                layer.m_output.reset();
                layer.m_dweights.reset();
                layer.m_dbiases.reset();
                layer.m_dinputs.reset();
                }
                
            // Clear all activation matrices
            for (auto& activation : m_activations) {
                activation.m_inputs.reset();
                activation.m_output.reset();
            }
        }

        void cleanup() {
            for (auto& layer : m_layers) {
                layer.reset();  // Clears intermediate inputs/outputs
            }
            for (auto& activation : m_activations) {
                activation.reset();  // Clears stored inputs/outputs
            }
        }

        void predict(double* input_data, double* output_data) {

            if (input_data == nullptr) {
                std::cerr << "Error: input_data is null" << std::endl;
                return;
            }
            const size_t num_elements = static_cast<size_t>(m_input_dim) * static_cast<size_t>(m_batch_size);
            if (num_elements == 0) {
                std::cerr << "Error: input_data has zero elements" << std::endl;
                return;
            }

            arma::mat inputs(input_data, m_input_dim, m_batch_size, true);
            inputs = inputs.t(); // Transpose to match the expected input shape

            for (int i = 0; i < m_layers.size() - 1; ++i) {
                m_layers[i].forward(inputs);
                m_activations[i].forward(m_layers[i].m_output);

                // set inputs for next layer
                inputs = m_activations[i].m_output;
                // print check
            }

            // Forward pass through the last layer
            m_layers.back().forward(inputs);


            // output is in m_activations[m_activations.size() - 1].m_output
            arma::mat& output = m_layers.back().m_output;
            if (output_data == nullptr) {
                std::cerr << "Error: output_data is null" << std::endl;
                return;
            }

            // Convert Armadillo matrix to double* (copy data)
            std::memcpy(output_data, output.memptr(), output.n_elem * sizeof(double));
        }


        void train(double* target_data) {
            //cleanup(); // Reset all layers and activations before training
            // Convert target_data to arma::mat
            arma::mat expected_output(target_data, m_layers.back().m_output.n_rows, m_layers.back().m_output.n_cols, true, false);

            arma::mat d_loss = derivative_mse_loss(m_layers.back().m_output, expected_output);
            
            m_layers.back().backward(d_loss);
            arma::mat d_act;
            
            for (int i = m_layers.size() - 2; i >= 0; --i) {
                d_act = m_activations[i].backward(m_layers[i + 1].m_dinputs);
                m_layers[i].backward(d_act);
            }

            // Update weights and biases
            optimizer.pre_update_params();
            for (int i = 0; i < m_layers.size(); ++i) {
                optimizer.update(m_layers[i]);
            }
        }

        bool save_model(const std::string& dirname) {
            return write_model(dirname, m_layers, m_input_dim, m_output_dim, m_hidden_dim, m_layers.size(), m_batch_size, m_nn_type);
        }

        uint32_t randomize_weights(std::vector<LayerDense>& layers) {
            for (auto& layer : layers) {
                layer.m_weights.randu();
                layer.m_biases.randu();
            }
            
            return 0;
        }
        
};

extern "C" {
    std::vector<std::unique_ptr<NeuralNetwork>> nn_online_instances; // id 0
    std::vector<std::unique_ptr<NeuralNetwork>> nn_target_instances; // id 1

    // vector of nn for RND -> predictor
    std::vector<std::unique_ptr<NeuralNetwork>> nn_rnd_instances; // id 2
    // vector of nn for RND -> target
    std::vector<std::unique_ptr<NeuralNetwork>> nn_rnd_target_instances; // id 3

    uint32_t init_nn(uint32_t input_dim, uint32_t output_dim, uint32_t hidden_dim, 
                 uint32_t num_m_layers, uint32_t batch_size, uint32_t nn_type) {
        // print intializing nn
        std::cout << "Initializing neural network with input_dim \n<" << input_dim 
                  << ", output_dim: " << output_dim 
                  << ", hidden_dim: " << hidden_dim 
                  << ", num_m_layers: " << num_m_layers 
                  << ", batch_size: " << batch_size
                  << ", nn_type (0=online, 1=target): " << nn_type
                  << ">" << std::endl;

        if (nn_type == 0) {
            nn_online_instances.push_back(std::make_unique<NeuralNetwork>(input_dim, output_dim, hidden_dim, num_m_layers, batch_size, nn_type));
            // print out online and target instances
            std::cout << "Online instances: " << nn_online_instances.size() << std::endl;
            std::cout << "Target instances: " << nn_target_instances.size() << std::endl;
            return nn_online_instances.size() - 1;
        } else if (nn_type == 1) {
            nn_target_instances.push_back(std::make_unique<NeuralNetwork>(input_dim, output_dim, hidden_dim, num_m_layers, batch_size, nn_type));
            // print out online and target instances
            std::cout << "Online instances: " << nn_online_instances.size() << std::endl;
            std::cout << "Target instances: " << nn_target_instances.size() << std::endl;
            return nn_target_instances.size() - 1;
        }
        else if (nn_type == 2) {
            nn_rnd_instances.push_back(std::make_unique<NeuralNetwork>(input_dim, output_dim, hidden_dim, num_m_layers, batch_size, nn_type));
            return nn_rnd_instances.size() - 1;
        } else if (nn_type == 3) {
            nn_rnd_target_instances.push_back(std::make_unique<NeuralNetwork>(input_dim, output_dim, hidden_dim, num_m_layers, batch_size, nn_type));
            return nn_rnd_target_instances.size() - 1;
        }
        else {
            std::cerr << "Error: Invalid neural network type" << std::endl;
            exit(1);
        }
    }

    void train_nn(uint32_t id, uint32_t nn_type, double* target_data) {
        if (nn_type == 0) {
            nn_online_instances[id]->train(target_data);
        }
        else if (nn_type == 1) {
            nn_target_instances[id]->train(target_data);
        }
        else if (nn_type == 2) {
            nn_rnd_instances[id]->train(target_data);
        }
        else if (nn_type == 3) {
            nn_rnd_target_instances[id]->train(target_data);
        }
        // If nn_type is not recognized, print an error message
        else {
            std::cerr << "Error: Invalid neural network type" << std::endl;
            exit(1);
        }
    }

    // Prediction function converts arma::mat to double*
    void predict_nn(uint32_t id, uint32_t nn_type, double* input_data, double* output_data) {
        if (nn_type == 0) {
            nn_online_instances[id]->predict(input_data, output_data);
        }
        else if (nn_type == 1) {
            nn_target_instances[id]->predict(input_data, output_data);
        }
        else if (nn_type == 2) {
            nn_rnd_instances[id]->predict(input_data, output_data);
        }
        else if (nn_type == 3) {
            nn_rnd_target_instances[id]->predict(input_data, output_data);
        }
    
        // If nn_type is not recognized, print an error message
        else {
            std::cerr << "Error: Invalid neural network type" << std::endl;
            exit(1);
        }
    }

    void update_target_nn(uint32_t online_nn_id, uint32_t target_nn_id) {
        // Check if the online and target neural networks exist
        if (online_nn_id >= nn_online_instances.size() || target_nn_id >= nn_target_instances.size()) {
            std::cerr << "Error: Invalid neural network ID" << std::endl;
            exit(1);
        }

        nn_target_instances[target_nn_id] = std::make_unique<NeuralNetwork>(*nn_online_instances[online_nn_id]);
    }

    bool save_nn_model(uint32_t id, uint32_t nn_type, const char* dirname) {
        if (nn_type == 0) {
            return nn_online_instances[id]->save_model(dirname);
        }
        else if (nn_type == 1) {
            return nn_target_instances[id]->save_model(dirname);
        }
        else if (nn_type == 2) {
            return nn_rnd_instances[id]->save_model(dirname);
        }
        else if (nn_type == 3) {
            return nn_rnd_target_instances[id]->save_model(dirname);
        }
        // If nn_type is not recognized, print an error message
        std::cerr << "Error: Invalid neural network type" << std::endl;
        return false;
    }

    uint32_t load_nn_model(const char* dirname, uint32_t nn_type) {
        try {
            NNInfo_metadata meta;
            std::vector<LayerDense> layers;
            
            if(!read_model(dirname, layers, meta)) {
                throw std::runtime_error("Failed to read model");
            }

            //auto& instances = (nn_type == 0) ? nn_online_instances : nn_target_instances;
            auto& instances = (nn_type == 0) ? nn_online_instances :
                              (nn_type == 1) ? nn_target_instances :
                              (nn_type == 2) ? nn_rnd_instances :
                              nn_rnd_target_instances;
            
            
            instances.push_back(std::make_unique<NeuralNetwork>(
                meta.input_dim,
                meta.output_dim,
                meta.hidden_dim,
                meta.num_m_layers,
                meta.batch_size,
                meta.nn_type
            ));

            auto& nn = *instances.back();
            nn.m_layers = std::move(layers);
    
            if(nn.m_activations.size() != nn.m_layers.size()) {
                nn.m_activations.clear();
                for (size_t i = 0; i < nn.m_layers.size(); ++i) {
                    nn.m_activations.emplace_back();
                }
            }

            return instances.size() - 1;
        } catch(const std::exception& e) {
            std::cerr << "Load NN Error: " << e.what() << std::endl;
            return UINT32_MAX;
        }
    }

    uint32_t randomize_weights(uint32_t id, uint32_t nn_type) {
        if (nn_type == 0) {
            return nn_online_instances[id]->randomize_weights(nn_online_instances[id]->m_layers);
        }
        else if (nn_type == 1) {
            return nn_target_instances[id]->randomize_weights(nn_target_instances[id]->m_layers);
        }
        else if (nn_type == 2) {
            return nn_rnd_instances[id]->randomize_weights(nn_rnd_instances[id]->m_layers);
        }
        else if (nn_type == 3) {
            return nn_rnd_target_instances[id]->randomize_weights(nn_rnd_target_instances[id]->m_layers);
        }
        // If nn_type is not recognized, print an error message
        std::cerr << "Error: Invalid neural network type" << std::endl;
        exit(1);
    }

    void reset_episode(uint32_t id, uint32_t nn_type) {
        if (nn_type == 0) {
            nn_online_instances[id]->reset_episode();
        }
        else if (nn_type == 1) {
            nn_target_instances[id]->reset_episode();
        }
        else if (nn_type == 2) {
            nn_rnd_instances[id]->reset_episode();
        }
        else if (nn_type == 3) {
            nn_rnd_target_instances[id]->reset_episode();
        }
        // If nn_type is not recognized, print an error message
        else {
            std::cerr << "Error: Invalid neural network type" << std::endl;
            exit(1);
        }
    }



}