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