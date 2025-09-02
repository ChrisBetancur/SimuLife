#ifndef IO_FRONTEND_H
#define IO_FRONTEND_H

namespace IO_FRONTEND {

    struct RND_Params {
        int DQN_INPUT_DIM;
        int DQN_OUTPUT_DIM;
        int DQN_HIDDEN_DIM;
        int DQN_NUM_LAYERS;
        int DQN_BATCH_SIZE;
    };

    struct DQN_Params {
        int RND_INPUT_DIM;
        int RND_OUTPUT_DIM;
        int RND_HIDDEN_DIM;
        int RND_NUM_LAYERS;
        int RND_BATCH_SIZE;
    };

    bool parse_rnd_params(const std::string& param_file_path, RND_Params& rnd_params); // parse RND hyperparam file, return success or not

    bool parse_dqn_params(const std::string& param_file_path, DQN_Params& dqn_params); // parse DQN hyperparam file, return success or not

    struct BoltzmannPolicy_Params {
        double initial_temp; 
        double decay_rate;
        double min_temp;
        double decay_interval;
    };

    bool parse_boltzmann_params(const std::string& param_file_path, BoltzmannPolicy_Params& bolzmann_params); // parse DQN hyperparam file, return success or not

    bool parse_buffer_capacity(const std::string& param_file_path, int& capacity);
}

#endif