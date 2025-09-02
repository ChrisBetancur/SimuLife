#include <io_frontend.h>

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
        if (line.find("RND_req_specs") != std::string::npos) {
            current_spec = "RND_req_specs";
            continue;
        }
        if (line.find("DQN_req_specs") != std::string::npos) {
            current_spec = "DQN_req_specs";
            continue;
        }
        if (line.find("BoltzmannPolicy_specs") != std::string::npos) {
            current_spec = "BoltzmannPolicy_specs";
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
                if (current_spec == "RND_req_specs" && std::is_same<T, IO_FRONTEND::RND_Params>::value) {
                    if (key == "RND_INPUT_DIM") params.RND_INPUT_DIM = std::stoi(value_str);
                    else if (key == "RND_OUTPUT_DIM") params.RND_OUTPUT_DIM = std::stoi(value_str);
                    else if (key == "RND_HIDDEN_DIM") params.RND_HIDDEN_DIM = std::stoi(value_str);
                    else if (key == "RND_NUM_LAYERS") params.RND_NUM_LAYERS = std::stoi(value_str);
                    else if (key == "RND_BATCH_SIZE") params.RND_BATCH_SIZE = std::stoi(value_str);
                } else if (current_spec == "DQN_req_specs" && std::is_same<T, IO_FRONTEND::DQN_Params>::value) {
                    if (key == "DQN_INPUT_DIM") params.DQN_INPUT_DIM = std::stoi(value_str);
                    else if (key == "DQN_OUTPUT_DIM") params.DQN_OUTPUT_DIM = std::stoi(value_str);
                    else if (key == "DQN_HIDDEN_DIM") params.DQN_HIDDEN_DIM = std::stoi(value_str);
                    else if (key == "DQN_NUM_LAYERS") params.DQN_NUM_LAYERS = std::stoi(value_str);
                    else if (key == "DQN_BATCH_SIZE") params.DQN_BATCH_SIZE = std::stoi(value_str);
                } else if (current_spec == "BoltzmannPolicy_specs" && std::is_same<T, IO_FRONTEND::BoltzmannPolicy_Params>::value) {
                    if (key == "initial_temp") params.initial_temp = std::stod(value_str);
                    else if (key == "decay_rate") params.decay_rate = std::stod(value_str);
                    else if (key == "min_temp") params.min_temp = std::stod(value_str);
                    else if (key == "decay_interval") params.decay_interval = std::stod(value_str);
                }
            }
        }
        
        // Check for standalone key-value pairs
        if (line.find("REPLAY_BUFFER_CAPACITY") != std::string::npos) {
            size_t equals_pos = line.find('=');
            if (equals_pos != std::string::npos) {
                std::string key = trim(line.substr(0, equals_pos));
                std::string value_str = trim(line.substr(equals_pos + 1));
                if (!value_str.empty() && value_str.back() == ';') {
                    value_str.pop_back();
                }
                if (key == "REPLAY_BUFFER_CAPACITY" && std::is_same<T, int>::value) {
                    params = std::stoi(value_str);
                }
            }
        }

    }
}


namespace IO_FRONTEND {

    bool parse_rnd_params(const std::string& param_file_path, RND_Params& rnd_params) {
        try {
            parse_params(param_file_path, rnd_params);
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Error parsing RND parameters: " << e.what() << std::endl;
            return false;
        }
    }

    bool parse_dqn_params(const std::string& param_file_path, DQN_Params& dqn_params) {
        try {
            parse_params(param_file_path, dqn_params);
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Error parsing DQN parameters: " << e.what() << std::endl;
            return false;
        }
    }

    bool parse_boltzmann_params(const std::string& param_file_path, BoltzmannPolicy_Params& boltzmann_params) {
        try {
            parse_params(param_file_path, boltzmann_params);
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Error parsing Boltzmann parameters: " << e.what() << std::endl;
            return false;
        }
    }

    bool parse_buffer_capacity(const std::string& param_file_path, int& capacity) {
        try {
            parse_params(param_file_path, capacity);
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Error parsing buffer capacity: " << e.what() << std::endl;
            return false;
        }
    }

} // namespace IO_FRONTEND