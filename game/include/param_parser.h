#ifndef PARAM_PARSER_H
#define PARAM_PARSER_H

#include <vector>

struct BoltzmannPolicy_specs {
    double initial_temp; 
    double decay_rate;
    double min_temp;
    double decay_interval;
};

struct DQN_req_specs {
    int DQN_INPUT_DIM; // 4 genome + 1 energy level + 2 vision (food count and is_wall) + 1 is_eating
    int DQN_OUTPUT_DIM; // Assuming 4 actions: UP, DOWN, LEFT, RIGHT
    int DQN_HIDDEN_DIM; // Hidden dimension for DQN networks
    int DQN_NUM_LAYERS; // Number of layers for DQN networks
    int DQN_BATCH_SIZE;
};

struct RND_req_specs {
    int RND_INPUT_DIM; // 4 genome + 1 energy level + 6 food rates
    int RND_OUTPUT_DIM; // Assuming 64 outputs for RND
    int RND_HIDDEN_DIM; // Hidden dimension for RND networks
    int RND_NUM_LAYERS; // Number of layers for RND networks
    int RND_BATCH_SIZE; // Batch size for RND training
};



#endif