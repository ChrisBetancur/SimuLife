#include <layer_dense.h>
#include <activation.h>
#include <optimizer.h>
#include <loss_utils.h>
#include <armadillo>
#include <vector>
#include <memory>
#include <io.h>

RND_Params rnd_params;
DQN_Params dqn_params;

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

        std::ofstream m_log_file;

        NeuralNetwork(uint32_t input_dim, uint32_t output_dim, uint32_t hidden_dim, 
                    uint32_t num_m_layers, uint32_t batch_size, uint32_t nn_type, 
                    double initial_lr, double beta1, double beta2, 
                    double eps, int max_steps, double min_lr) :
            optimizer(initial_lr, beta1, beta2, eps, 0.0, max_steps, min_lr),
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

            std::string filename;
            switch (m_nn_type) {
                case 0: filename = "online_system.log"; break;
                case 2: filename = "rnd_predictor_system.log"; break;
                default: filename = ""; break; // No log for target networks
            }

            if (!filename.empty()) {
                std::filesystem::create_directory("logs");
                std::string full_path = "logs/" + filename; 
                
                // Add this line to ensure a clean file for logging
                if (std::filesystem::exists(full_path)) {
                    std::filesystem::remove(full_path);
                }

                m_log_file.open(full_path, std::ios::app);

                if (!m_log_file.is_open()) {
                    std::cerr << "Error: Could not open log file " << full_path << ". Reason: " << strerror(errno) << std::endl;
                    // It is safer to exit or throw here, as logging won't work otherwise.
                } else {
                    std::cout << "Successfully opened log file: " << full_path << std::endl;
                }
            }

        }

        ~NeuralNetwork() {
            if (m_log_file.is_open()) {
                m_log_file.close();
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


            // verify tgat weights and biases are not NaN
            for (const auto& layer : m_layers) {
                if (layer.m_weights.has_nan() || layer.m_biases.has_nan()) {
                    std::cerr << "Error: Layer weights or biases contain NaN values." << std::endl;
                    exit(1);
                }
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

        void predict(double* input_data, double* output_data, uint32_t batch_size) {

            if (input_data == nullptr) {
                std::cerr << "Error: input_data is null" << std::endl;
                return;
            }
            const size_t num_elements = static_cast<size_t>(m_input_dim) * static_cast<size_t>(batch_size);
            if (num_elements == 0) {
                std::cerr << "Error: input_data has zero elements" << std::endl;
                return;
            }

            arma::mat inputs(input_data, m_input_dim, batch_size, true);
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

            // check if output has NaN or infinite values
            if (output.has_nan() || output.has_inf()) {
                std::cerr << "Error: Output contains NaN or infinite values." << std::endl;
                std::cerr << "Output matrix: " << output << std::endl;
                exit(1);
            }

            // Convert Armadillo matrix to double* (copy data)
            std::memcpy(output_data, output.memptr(), output.n_elem * sizeof(double));
        }


        void train(double* input_data, double* target_data) {
            arma::mat inputs(input_data, m_input_dim, m_batch_size, true);
            inputs = inputs.t(); // Transpose to match the expected input shape

            for (int i = 0; i < m_layers.size() - 1; ++i) {
                m_layers[i].forward(inputs);
                m_activations[i].forward(m_layers[i].m_output);

                inputs = m_activations[i].m_output;
            }

            // Forward pass through the last layer
            m_layers.back().forward(inputs);

            arma::mat expected_output(target_data, m_layers.back().m_output.n_rows, m_layers.back().m_output.n_cols, true, false);

            //double loss = mse_loss(m_layers.back().m_output, expected_output);

            const double huber_delta = 1.0;
            double loss = huber_loss(m_layers.back().m_output, expected_output, huber_delta);

            double reg_val = 0.0;
            for (const auto& layer : m_layers) {
                reg_val += regularization_loss(layer);  // your function
            }
            double total_loss = loss + reg_val;

            if (m_log_file.is_open() && (m_nn_type == 0 || m_nn_type == 2)) {
                m_log_file << total_loss << std::endl;
                m_log_file.flush();
            }

            // if nn_type is 0 write hello to log file
            /*if (m_log_file.is_open() && m_log_file.good()) {
                // If nn_type is 0 (online) or 2 (RND predictor) log the loss
                m_log_file << std::setprecision(5) << loss << std::endl;
                m_log_file.flush();
            }*/

            //arma::mat d_loss = derivative_mse_loss(m_layers.back().m_output, expected_output);
            arma::mat d_loss = derivative_huber_loss(m_layers.back().m_output, expected_output, huber_delta);

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

            /*if (m_log_file.is_open() && (m_nn_type == 0 || m_nn_type == 2)) {
                m_log_file << loss << std::endl;
                m_log_file.flush();
            }*/

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

    uint32_t parse_nn_params() {
        std::cout << "Hyperparameter Initilization for Neural Network" << std::endl;
        
        bool status = parse_dqn_params("../neural_network", dqn_params);
        
        if (status == false)
            return 1;

        std::cout << "DQN Initial Learning Rate: " << dqn_params.LR_INITIAL << std::endl;
        std::cout << "DQN BETA1: " << dqn_params.BETA1 << std::endl;
        std::cout << "DQN BETA2: " << dqn_params.BETA2<< std::endl;
        std::cout << "DQN EPS: " << dqn_params.EPS << std::endl;
        std::cout << "DQN Max Training Steps: " << dqn_params.max_training_steps << std::endl;
        std::cout << "DQN Minimum Learning Rate: " << dqn_params.min_learning_rate << std::endl << std::endl;

        status = parse_rnd_params("../neural_network", rnd_params);
        if (status == false)
            return 1;

        std::cout << "RND Initial Learning Rate: " << rnd_params.LR_INITIAL << std::endl;
        std::cout << "RND BETA1: " << rnd_params.BETA1 << std::endl;
        std::cout << "RND BETA2: " << rnd_params.BETA2<< std::endl;
        std::cout << "RND EPS: " << rnd_params.EPS << std::endl;
        std::cout << "RND Max Training Steps: " << rnd_params.max_training_steps << std::endl;
        std::cout << "RND Minimum Training Steps: " << rnd_params.min_learning_rate << std::endl;

        std::cout << "End of Hyperparameter Initlization for Neural Network" << std::endl << std::endl;

        return 0;
    }

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
            nn_online_instances.push_back(std::make_unique<NeuralNetwork>(input_dim, output_dim, hidden_dim, num_m_layers, batch_size, nn_type, 
                dqn_params.LR_INITIAL, dqn_params.BETA1, dqn_params.BETA2, dqn_params.EPS, dqn_params.max_training_steps, dqn_params.min_learning_rate));
            // print out online and target instances
            std::cout << "Online instances: " << nn_online_instances.size() << std::endl;
            std::cout << "Target instances: " << nn_target_instances.size() << std::endl;
            return nn_online_instances.size() - 1;
        } else if (nn_type == 1) {
            nn_target_instances.push_back(std::make_unique<NeuralNetwork>(input_dim, output_dim, hidden_dim, num_m_layers, batch_size, nn_type, 
                dqn_params.LR_INITIAL, dqn_params.BETA1, dqn_params.BETA2, dqn_params.EPS, dqn_params.max_training_steps, dqn_params.min_learning_rate));
            // print out online and target instances
            std::cout << "Online instances: " << nn_online_instances.size() << std::endl;
            std::cout << "Target instances: " << nn_target_instances.size() << std::endl;
            return nn_target_instances.size() - 1;
        }
        else if (nn_type == 2) {
            nn_rnd_instances.push_back(std::make_unique<NeuralNetwork>(input_dim, output_dim, hidden_dim, num_m_layers, batch_size, nn_type, 
                rnd_params.LR_INITIAL, rnd_params.BETA1, rnd_params.BETA2, rnd_params.EPS, rnd_params.max_training_steps, rnd_params.min_learning_rate));
            return nn_rnd_instances.size() - 1;
        } else if (nn_type == 3) {
            nn_rnd_target_instances.push_back(std::make_unique<NeuralNetwork>(input_dim, output_dim, hidden_dim, num_m_layers, batch_size, nn_type, 
                rnd_params.LR_INITIAL, rnd_params.BETA1, rnd_params.BETA2, rnd_params.EPS, rnd_params.max_training_steps, rnd_params.min_learning_rate));
            return nn_rnd_target_instances.size() - 1;
        }
        else {
            std::cerr << "Error: Invalid neural network type" << std::endl;
            exit(1);
        }
    }

    void train_nn(uint32_t id, uint32_t nn_type, double* input_data, double* target_data, uint32_t batch_size) {
        if (nn_type == 0) {
            nn_online_instances[id]->train(input_data, target_data);
        }
        else if (nn_type == 1) {
            nn_target_instances[id]->train(input_data, target_data);
        }
        else if (nn_type == 2) {
            nn_rnd_instances[id]->train(input_data, target_data);
        }
        else if (nn_type == 3) {
            nn_rnd_target_instances[id]->train(input_data, target_data);
        }
        // If nn_type is not recognized, print an error message
        else {
            std::cerr << "Error: Invalid neural network type" << std::endl;
            exit(1);
        }
    }

    // Prediction function converts arma::mat to double*
    void predict_nn(uint32_t id, uint32_t nn_type, double* input_data, double* output_data, uint32_t batch_size) {
        if (nn_type == 0) {
            nn_online_instances[id]->predict(input_data, output_data, batch_size);
        }
        else if (nn_type == 1) {
            nn_target_instances[id]->predict(input_data, output_data, batch_size);
        }
        else if (nn_type == 2) {
            nn_rnd_instances[id]->predict(input_data, output_data, batch_size);
        }
        else if (nn_type == 3) {
            nn_rnd_target_instances[id]->predict(input_data, output_data, batch_size);
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
            
            /*
            instances.push_back(std::make_unique<NeuralNetwork>(
                meta.input_dim,
                meta.output_dim,
                meta.hidden_dim,
                meta.num_m_layers,
                meta.batch_size,
                meta.nn_type
            ));*/

            if (meta.nn_type == 0) {
                nn_online_instances.push_back(std::make_unique<NeuralNetwork>(
                    meta.input_dim,
                    meta.output_dim,
                    meta.hidden_dim,
                    meta.num_m_layers,
                    meta.batch_size,
                    meta.nn_type,
                    dqn_params.LR_INITIAL, dqn_params.BETA1, dqn_params.BETA2, dqn_params.EPS, dqn_params.max_training_steps, dqn_params.min_learning_rate));
            } else if (meta.nn_type == 1) {
                nn_target_instances.push_back(std::make_unique<NeuralNetwork>(
                    meta.input_dim,
                    meta.output_dim,
                    meta.hidden_dim,
                    meta.num_m_layers,
                    meta.batch_size,
                    meta.nn_type,
                    dqn_params.LR_INITIAL, dqn_params.BETA1, dqn_params.BETA2, dqn_params.EPS, dqn_params.max_training_steps, dqn_params.min_learning_rate));
            } else if (meta.nn_type == 2) {
                nn_rnd_instances.push_back(std::make_unique<NeuralNetwork>(
                    meta.input_dim,
                    meta.output_dim,
                    meta.hidden_dim,
                    meta.num_m_layers,
                    meta.batch_size,
                    meta.nn_type,
                    rnd_params.LR_INITIAL, rnd_params.BETA1, rnd_params.BETA2, rnd_params.EPS, rnd_params.max_training_steps, rnd_params.min_learning_rate));
            } else if (meta.nn_type == 3) {
                nn_rnd_target_instances.push_back(std::make_unique<NeuralNetwork>(
                    meta.input_dim,
                    meta.output_dim,
                    meta.hidden_dim,
                    meta.num_m_layers,
                    meta.batch_size,
                    meta.nn_type,
                    rnd_params.LR_INITIAL, rnd_params.BETA1, rnd_params.BETA2, rnd_params.EPS, rnd_params.max_training_steps, rnd_params.min_learning_rate));
            } else {
                throw std::runtime_error("Invalid neural network type");
            }

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




}