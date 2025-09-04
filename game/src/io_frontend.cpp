#include <io_frontend.h>
#include <fstream>
#include <string>
#include <stdexcept>
#include <iostream>

// This function removes leading/trailing whitespace.
inline std::string trim(const std::string& str) {
    size_t first = str.find_first_not_of(" \t\n\r");
    if (std::string::npos == first) {
        return str;
    }
    size_t last = str.find_last_not_of(" \t\n\r");
    return str.substr(first, (last - first + 1));
}

// Helper function to print RND_Params
void print_rnd_params(const IO_FRONTEND::RND_Params& params) {
    std::cout << "Parsed RND Parameters:" << std::endl;
    std::cout << "  RND_INPUT_DIM: " << params.RND_INPUT_DIM << std::endl;
    std::cout << "  RND_OUTPUT_DIM: " << params.RND_OUTPUT_DIM << std::endl;
    std::cout << "  RND_HIDDEN_DIM: " << params.RND_HIDDEN_DIM << std::endl;
    std::cout << "  RND_NUM_LAYERS: " << params.RND_NUM_LAYERS << std::endl;
    std::cout << "  RND_BATCH_SIZE: " << params.RND_BATCH_SIZE << std::endl;
    std::cout << "------------------------" << std::endl;
}

// Helper function to print DQN_Params
void print_dqn_params(const IO_FRONTEND::DQN_Params& params) {
    std::cout << "Parsed DQN Parameters:" << std::endl;
    std::cout << "  DQN_INPUT_DIM: " << params.DQN_INPUT_DIM << std::endl;
    std::cout << "  DQN_OUTPUT_DIM: " << params.DQN_OUTPUT_DIM << std::endl;
    std::cout << "  DQN_HIDDEN_DIM: " << params.DQN_HIDDEN_DIM << std::endl;
    std::cout << "  DQN_NUM_LAYERS: " << params.DQN_NUM_LAYERS << std::endl;
    std::cout << "  DQN_BATCH_SIZE: " << params.DQN_BATCH_SIZE << std::endl;
    std::cout << "------------------------" << std::endl;
}

// Helper function to print BoltzmannPolicy_Params
void print_boltzmann_params(const IO_FRONTEND::BoltzmannPolicy_Params& params) {
    std::cout << "Parsed Boltzmann Policy Parameters:" << std::endl;
    std::cout << "  initial_temp: " << params.initial_temp << std::endl;
    std::cout << "  decay_rate: " << params.decay_rate << std::endl;
    std::cout << "  min_temp: " << params.min_temp << std::endl;
    std::cout << "  decay_interval: " << params.decay_interval << std::endl;
    std::cout << "------------------------" << std::endl;
}

// Function to parse parameters for RND_Params
void parse_rnd_params_impl(const std::string& file_path, IO_FRONTEND::RND_Params& params) {
    std::ifstream file(file_path);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + file_path);
    }

    std::string line;
    bool in_rnd_spec = false;
    while (std::getline(file, line)) {
        line = trim(line);
        if (line.empty() || line.front() == '#') {
            continue;
        }

        if (line.find("RND_req_specs") != std::string::npos) {
            in_rnd_spec = true;
            continue;
        }
        if (in_rnd_spec && line.find('}') != std::string::npos) {
            in_rnd_spec = false;
            continue;
        }

        if (in_rnd_spec) {
            size_t equals_pos = line.find('=');
            if (equals_pos != std::string::npos) {
                std::string key = trim(line.substr(0, equals_pos));
                std::string value_str = trim(line.substr(equals_pos + 1));
                
                if (!value_str.empty() && value_str.back() == ';') {
                    value_str.pop_back();
                }

                if (key == "RND_INPUT_DIM") params.RND_INPUT_DIM = std::stoi(value_str);
                else if (key == "RND_OUTPUT_DIM") params.RND_OUTPUT_DIM = std::stoi(value_str);
                else if (key == "RND_HIDDEN_DIM") params.RND_HIDDEN_DIM = std::stoi(value_str);
                else if (key == "RND_NUM_LAYERS") params.RND_NUM_LAYERS = std::stoi(value_str);
                else if (key == "RND_BATCH_SIZE") params.RND_BATCH_SIZE = std::stoi(value_str);
            }
        }
    }
}

// Function to parse parameters for DQN_Params
void parse_dqn_params_impl(const std::string& file_path, IO_FRONTEND::DQN_Params& params) {
    std::ifstream file(file_path);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + file_path);
    }

    std::string line;
    bool in_dqn_spec = false;
    while (std::getline(file, line)) {
        line = trim(line);
        if (line.empty() || line.front() == '#') {
            continue;
        }

        if (line.find("DQN_req_specs") != std::string::npos) {
            in_dqn_spec = true;
            continue;
        }
        if (in_dqn_spec && line.find('}') != std::string::npos) {
            in_dqn_spec = false;
            continue;
        }

        if (in_dqn_spec) {
            size_t equals_pos = line.find('=');
            if (equals_pos != std::string::npos) {
                std::string key = trim(line.substr(0, equals_pos));
                std::string value_str = trim(line.substr(equals_pos + 1));
                
                if (!value_str.empty() && value_str.back() == ';') {
                    value_str.pop_back();
                }

                if (key == "DQN_INPUT_DIM") params.DQN_INPUT_DIM = std::stoi(value_str);
                else if (key == "DQN_OUTPUT_DIM") params.DQN_OUTPUT_DIM = std::stoi(value_str);
                else if (key == "DQN_HIDDEN_DIM") params.DQN_HIDDEN_DIM = std::stoi(value_str);
                else if (key == "DQN_NUM_LAYERS") params.DQN_NUM_LAYERS = std::stoi(value_str);
                else if (key == "DQN_BATCH_SIZE") params.DQN_BATCH_SIZE = std::stoi(value_str);
            }
        }
    }
}

// Function to parse parameters for BoltzmannPolicy_Params
void parse_boltzmann_params_impl(const std::string& file_path, IO_FRONTEND::BoltzmannPolicy_Params& params) {
    std::ifstream file(file_path);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + file_path);
    }

    std::string line;
    bool in_boltzmann_spec = false;
    while (std::getline(file, line)) {
        line = trim(line);
        if (line.empty() || line.front() == '#') {
            continue;
        }

        if (line.find("BoltzmannPolicy_specs") != std::string::npos) {
            in_boltzmann_spec = true;
            continue;
        }
        if (in_boltzmann_spec && line.find('}') != std::string::npos) {
            in_boltzmann_spec = false;
            continue;
        }

        if (in_boltzmann_spec) {
            size_t equals_pos = line.find('=');
            if (equals_pos != std::string::npos) {
                std::string key = trim(line.substr(0, equals_pos));
                std::string value_str = trim(line.substr(equals_pos + 1));
                
                if (!value_str.empty() && value_str.back() == ';') {
                    value_str.pop_back();
                }

                if (key == "initial_temp") params.initial_temp = std::stod(value_str);
                else if (key == "decay_rate") params.decay_rate = std::stod(value_str);
                else if (key == "min_temp") params.min_temp = std::stod(value_str);
                else if (key == "decay_interval") params.decay_interval = std::stod(value_str);
            }
        }
    }
}

// Function to parse buffer capacity
void parse_buffer_capacity_impl(const std::string& file_path, int& capacity) {
    std::ifstream file(file_path);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + file_path);
    }

    std::string line;
    while (std::getline(file, line)) {
        line = trim(line);
        if (line.empty() || line.front() == '#') {
            continue;
        }

        if (line.find("REPLAY_BUFFER_CAPACITY") != std::string::npos) {
            size_t equals_pos = line.find('=');
            if (equals_pos != std::string::npos) {
                std::string key = trim(line.substr(0, equals_pos));
                std::string value_str = trim(line.substr(equals_pos + 1));
                if (!value_str.empty() && value_str.back() == ';') {
                    value_str.pop_back();
                }
                if (key == "REPLAY_BUFFER_CAPACITY") {
                    capacity = std::stoi(value_str);
                    // Found the value, no need to continue parsing
                    return;
                }
            }
        }
    }
}

namespace IO_FRONTEND {

    bool parse_rnd_params(const std::string& param_file_path, RND_Params& rnd_params) {
        try {
            parse_rnd_params_impl(param_file_path, rnd_params);
            // Print the parameters after successful parsing
            //print_rnd_params(rnd_params);
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Error parsing RND parameters: " << e.what() << std::endl;
            return false;
        }
    }

    bool parse_dqn_params(const std::string& param_file_path, DQN_Params& dqn_params) {
        try {
            parse_dqn_params_impl(param_file_path, dqn_params);
            // Print the parameters after successful parsing
            //print_dqn_params(dqn_params);
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Error parsing DQN parameters: " << e.what() << std::endl;
            return false;
        }
    }

    bool parse_boltzmann_params(const std::string& param_file_path, BoltzmannPolicy_Params& boltzmann_params) {
        try {
            parse_boltzmann_params_impl(param_file_path, boltzmann_params);
            // Print the parameters after successful parsing
            //print_boltzmann_params(boltzmann_params);
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Error parsing Boltzmann parameters: " << e.what() << std::endl;
            return false;
        }
    }

    bool parse_buffer_capacity(const std::string& param_file_path, int& capacity) {
        try {
            parse_buffer_capacity_impl(param_file_path, capacity);
            // Print the parameters after successful parsing
            //std::cout << "Parsed Buffer Capacity: " << capacity << std::endl;
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Error parsing buffer capacity: " << e.what() << std::endl;
            return false;
        }
    }

} // namespace IO_FRONTEND