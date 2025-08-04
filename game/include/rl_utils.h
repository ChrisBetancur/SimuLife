#ifndef RL_UTILS_H
#define RL_UTILS_H

#include <organism.h>
#include <map>
#include <sprites.h>
#include <map.h>
#include <random>
#include <rl_utils.h>
#include <nn_api.h>

#define MAX_ENERGY 100.0f

struct State {
    Genome genome;
    double energy_lvl;
    std::tuple<int, bool> vision;
};

struct Action {
    Direction direction;
};

enum class PolicyType {
    EPSILON_GREEDY,
    BOLTZMANN
};

double computeExtrinsicReward(State state, Action action);

double computeReward(State state, Action action, std::vector<double> food_rates, uint32_t organism_sector, bool enable_rnd = false);

double* prepareInputData(State state, bool is_RND, std::vector<double> food_rates, uint32_t organism_sector);

#endif // RL_UTILS_H