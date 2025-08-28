#include <io.h>
#include <layer_dense.h>

#include <fstream>
#include <vector>
#include <string>
#include <cstdint>
#include <armadillo>

bool write_model(const std::string& dirname, const std::vector<LayerDense>& layers, uint32_t input_dim, uint32_t output_dim, uint32_t hidden_dim, 
    uint32_t num_m_layers, uint32_t batch_size, uint32_t nn_type) {
    //layers->clear();
    //layers->resize(num_m_layers);
    // Validate input
    if (layers.empty()) {
        throw std::runtime_error("No layers to write");
        return false;
    }

    // Create directory if it doesn't exist
    std::filesystem::create_directories(dirname);

    std::error_code dir_error;
    std::filesystem::create_directories(dirname, dir_error);
    if (dir_error) {
        throw std::runtime_error("Failed to create directory: " + dirname + " - " + dir_error.message());
    }
    if (!std::filesystem::is_directory(dirname)) {
        throw std::runtime_error("Path is not a directory: " + dirname);
    }

    // write NN auxilary info to file

    // init out
    std::ofstream out;

    out.open(dirname + "/nn_info.bin", std::ios::binary);
    if (!out) {
        throw std::runtime_error("Cannot open file: " + dirname + "/nn_info.bin");
    }

    // Write num layers, batch size and NN type


    // Write input,output and hidden dims
    // Write num layers, batch size and NN type
    out.write(reinterpret_cast<const char*>(&input_dim), sizeof(input_dim));
    out.write(reinterpret_cast<const char*>(&output_dim), sizeof(output_dim));
    out.write(reinterpret_cast<const char*>(&hidden_dim), sizeof(hidden_dim));
    out.write(reinterpret_cast<const char*>(&num_m_layers), sizeof(num_m_layers));
    out.write(reinterpret_cast<const char*>(&batch_size), sizeof(batch_size));
    out.write(reinterpret_cast<const char*>(&nn_type), sizeof(nn_type));

    for (size_t i = 0; i < layers.size(); ++i) {
        std::ofstream out;
        const LayerDense& layer = layers[i];

        // Helper function to write layer data
        auto write_layer_data = [&](const std::string& filename, 
                                  const auto& matrix) {
            out.open(filename, std::ios::binary);
            if (!out) {
                throw std::runtime_error("Cannot open file: " + filename);
            }

            // Write matrix dimensions (rows, cols) instead of num_layers
            uint32_t rows = static_cast<uint32_t>(matrix.n_rows);
            uint32_t cols = static_cast<uint32_t>(matrix.n_cols);
            out.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
            out.write(reinterpret_cast<const char*>(&cols), sizeof(cols));

            // Write matrix data
            out.write(reinterpret_cast<const char*>(matrix.memptr()),
                     matrix.n_elem * sizeof(typename std::decay_t<decltype(matrix)>::elem_type));
            
            out.close();
        };

        // Generate filenames
        std::string prefix = dirname + "/layer" + std::to_string(i);
        
        // Write weights and biases
        write_layer_data(prefix + "_weights.bin", layer.m_weights);
        write_layer_data(prefix + "_biases.bin", layer.m_biases);
        write_layer_data(prefix + "_velocity_weights.bin", layer.m_velocity_weights);
        write_layer_data(prefix + "_velocity_biases.bin", layer.m_velocity_biases);
    }
    return true;
}

bool read_model(const std::string& dirname, std::vector<LayerDense>& layers, NNInfo_metadata& nn_info) {
    namespace fs = std::filesystem;
    
    try {
        // 1. Validate model directory
        if (!fs::is_directory(dirname)) {
            throw std::runtime_error("Path is not a directory: " + dirname);
        }

        // 2. Check if metadata file exists
        if (!fs::exists(fs::path(dirname) / "nn_info.bin")) {
            throw std::runtime_error("Metadata file not found in directory: " + dirname);
        }

        // 2. Read metadata
        std::ifstream meta_in(fs::path(dirname) / "nn_info.bin", std::ios::binary);
        meta_in.read(reinterpret_cast<char*>(&nn_info), sizeof(NNInfo_metadata));
        
        // 3. Prepare layers vector
        layers.clear();
        layers.reserve(nn_info.num_m_layers);

        // 4. Initialize layers with proper dimensions
        for(uint32_t i = 0; i < nn_info.num_m_layers; ++i) {
            const uint32_t in_dim = (i == 0) ? nn_info.input_dim : nn_info.hidden_dim;
            const uint32_t out_dim = (i == nn_info.num_m_layers-1) ? nn_info.output_dim : nn_info.hidden_dim;
            
            layers.emplace_back(in_dim, out_dim, 0.0, 0.0001, 0.0, 0);
        }

        // 5. Load layer parameters
        // 5. Load layer parameters
        auto load_layer = [](LayerDense& layer, const std::string& prefix) {
            auto read_matrix = [](const std::string& path) -> arma::mat {
                std::ifstream in(path, std::ios::binary);
                if (!in) {
                    throw std::runtime_error("Cannot open file: " + path);
                }

                // Read matrix dimensions
                uint32_t rows, cols;
                in.read(reinterpret_cast<char*>(&rows), sizeof(rows));
                in.read(reinterpret_cast<char*>(&cols), sizeof(cols));

                // Create and fill matrix
                arma::mat matrix(rows, cols);
                in.read(reinterpret_cast<char*>(matrix.memptr()), 
                        rows * cols * sizeof(arma::mat::elem_type));
                
                return matrix;
            };

            layer.m_weights = read_matrix(prefix + "_weights.bin");
            layer.m_biases = read_matrix(prefix + "_biases.bin");
            layer.m_velocity_weights = read_matrix(prefix + "_velocity_weights.bin");
            layer.m_velocity_biases = read_matrix(prefix + "_velocity_biases.bin");
        };

        // 6. Load each layer
        for(uint32_t i = 0; i < nn_info.num_m_layers; ++i) {
            const auto prefix = (fs::path(dirname) / ("layer" + std::to_string(i))).string();
            load_layer(layers[i], prefix);
        }

        return true;
    } catch(const std::exception& e) {
        std::cerr << "Load Model Error: " << e.what() << std::endl;
        return false;
    }
}

// Helper function to trim whitespace from a string
std::string trim(const std::string& str) {
    size_t first = str.find_first_not_of(" \t\n\r");
    if (std::string::npos == first) {
        return str;
    }
    size_t last = str.find_last_not_of(" \t\n\r");
    return str.substr(first, (last - first + 1));
}

// Generic function to parse key-value pairs from the file
template<typename T>
void parse_params(const std::string& file_path, T& params) {
    std::ifstream file(file_path);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + file_path);
    }

    std::string line;
    std::string current_spec;
    while (std::getline(file, line)) {
        line = trim(line);
        if (line.empty() || line.front() == '#') {
            continue;
        }

        // Check for the start of a new spec block
        if (line.find("RND_specs") != std::string::npos) {
            current_spec = "RND_specs";
            continue;
        }
        if (line.find("DQN_specs") != std::string::npos) {
            current_spec = "DQN_specs";
            continue;
        }

        // If inside a spec block and not the end brace
        if (!current_spec.empty() && line.find('}') == std::string::npos) {
            size_t equals_pos = line.find('=');
            if (equals_pos != std::string::npos) {
                std::string key = trim(line.substr(0, equals_pos));
                std::string value_str = trim(line.substr(equals_pos + 1));
                
                // Remove trailing semicolon if present
                if (!value_str.empty() && value_str.back() == ';') {
                    value_str.pop_back();
                }

                // Check which struct we are populating
                if (current_spec == "RND_specs" && std::is_same<T, RND_Params>::value) {
                    if (key == "LR_INITIAL") params.LR_INITIAL = std::stod(value_str);
                    else if (key == "BETA1") params.BETA1 = std::stod(value_str);
                    else if (key == "BETA2") params.BETA2 = std::stod(value_str);
                    else if (key == "EPS") params.EPS = std::stod(value_str);
                    else if (key == "max_training_steps") params.max_training_steps = std::stoi(value_str);
                    else if (key == "min_learning_rate") params.min_learning_rate = std::stod(value_str);
                } else if (current_spec == "DQN_specs" && std::is_same<T, DQN_Params>::value) {
                    if (key == "LR_INITIAL") params.LR_INITIAL = std::stod(value_str);
                    else if (key == "BETA1") params.BETA1 = std::stod(value_str);
                    else if (key == "BETA2") params.BETA2 = std::stod(value_str);
                    else if (key == "EPS") params.EPS = std::stod(value_str);
                    else if (key == "max_training_steps") params.max_training_steps = std::stoi(value_str);
                    else if (key == "min_learning_rate") params.min_learning_rate = std::stod(value_str);
                }
            }
        }
    }
}

bool parse_rnd_params(const std::string& param_file_path, RND_Params& rnd_params) {
    namespace fs = std::filesystem;
    
    try {
        // Check if param file exists
        fs::path file_path = fs::path(param_file_path) / "nn_system.params";
        if (!fs::exists(file_path)) {
            throw std::runtime_error("NN System Params not found: " + file_path.string());
        }

        parse_params(file_path.string(), rnd_params);
        
        return true;
    } catch(const std::exception& e) {
        std::cerr << "Load NN Param Error: " << e.what() << std::endl;
        return false;
    }
}

bool parse_dqn_params(const std::string& param_file_path, DQN_Params& dqn_params) {
    namespace fs = std::filesystem;
    
    try {
        // Check if param file exists
        fs::path file_path = fs::path(param_file_path) / "nn_system.params";
        if (!fs::exists(file_path)) {
            throw std::runtime_error("NN System Params not found: " + file_path.string());
        }

        parse_params(file_path.string(), dqn_params);
        
        return true;
    } catch(const std::exception& e) {
        std::cerr << "Load NN Param Error: " << e.what() << std::endl;
        return false;
    }
}