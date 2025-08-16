#ifndef RL_UTILS_H
#define RL_UTILS_H

#include <organism.h>
#include <map>
#include <sprites.h>
#include <map.h>
#include <random>
#include <rl_utils.h>
#include <nn_api.h>
#include <cmath>

#define MAX_ENERGY 100.0f

#define DQN_INPUT_DIM 7 // 4 genome + 1 energy level + 2 vision (food count and is_wall)
#define DQN_OUTPUT_DIM 4 // Assuming 4 actions: UP, DOWN, LEFT, RIGHT
#define DQN_HIDDEN_DIM 64 // Hidden dimension for DQN networks
#define DQN_NUM_LAYERS 5 // Number of layers for DQN networks

#define DQN_ONLINE_ID 0 // ID for DQN online
#define DQN_TARGET_ID 1 // ID for DQN target

#define RND_INPUT_DIM 11 // 4 genome + 1 energy level + 6 food rates
#define RND_OUTPUT_DIM 64 // Assuming 64 outputs for RND
#define RND_HIDDEN_DIM 64 // Hidden dimension for RND networks
#define RND_NUM_LAYERS 5 // Number of layers for RND networks

#define RND_PREDICTOR_ID 2 // ID for RND predictor
#define RND_TARGET_ID 3 // ID for RND target

#define RND_BATCH_SIZE 32 // Batch size for RND training

inline double tanh_scale(double x, double amplitude, double sensitivity) {
    if (sensitivity <= 0.0) sensitivity = 1.0;
    return amplitude * std::tanh(x / sensitivity);
}

struct State {
    Genome genome;
    double energy_lvl;
    std::tuple<int, bool, int> vision; // Tuple: (food_count, is_wall, wall_distance)
    int food_count; // Number of food items in the vision
    bool is_new_episode; // Flag to indicate if this is a new episode
};

struct Action {
    Direction direction;
};

struct Transition {
    State state;
    Action action;
    float reward;
    State next_state;
    bool done; // Indicates if the episode has ended
};

enum class PolicyType {
    EPSILON_GREEDY,
    BOLTZMANN
};

double computeIntrinsicReward(int rnd_batch_size);

double computeExtrinsicReward(State state, Action action, bool hit_wall, int org_x, int org_y, Direction dir, int wall_pos_x = -1, int wall_pos_y = -1);

double computeReward(State state, Action action, std::vector<double> food_rates, uint32_t organism_sector, bool enable_rnd, bool hit_wall, int org_x, int org_y, Direction dir, int wall_pos_x = -1, int wall_pos_y = -1);

double* prepareInputData(State state, bool is_RND, std::vector<double> food_rates, uint32_t organism_sector);

#endif // RL_UTILS_H