#ifndef IO_H
#define IO_H
#include <armadillo>
#include <layer_dense.h>

struct NNInfo_metadata {
    uint32_t input_dim;
    uint32_t output_dim;
    uint32_t hidden_dim;
    uint32_t num_m_layers;
    uint32_t batch_size;
    uint32_t nn_type;
};

bool write_model(const std::string& dirname,
    const std::vector<LayerDense>& layers, uint32_t input_dim, uint32_t output_dim, uint32_t hidden_dim, 
    uint32_t num_m_layers, uint32_t batch_size, uint32_t nn_type);

bool read_model(const std::string& dirname,
    std::vector<LayerDense>& layers, NNInfo_metadata& nn_info);

struct RND_Params {
    double LR_INITIAL;
    double BETA1;
    double BETA2;
    double EPS;
    int max_training_steps;
    double min_learning_rate;
};

struct DQN_Params {
    double LR_INITIAL;
    double BETA1;
    double BETA2;
    double EPS;
    int max_training_steps;
    double min_learning_rate;
};

bool parse_rnd_params(const std::string& param_file_path, RND_Params& rnd_params); // parse RND hyperparam file, return success or not

bool parse_dqn_params(const std::string& param_file_path, DQN_Params& dqn_params); // parse DQN hyperparam file, return success or not


#endif