#include <rl_utils.h>
#include <stats.h>

double computeExtrinsicReward(State state, Action action) {
    const float WALL_PENALTY = -1.0f;    // Base penalty for wall collision
    const float FOOD_REWARD = 1.0f;      // Base reward for food
    double reward = 0.0f;
    bool wall_in_path = false;
    bool food_in_path = false;

    // Vision processing with distance weighting
    for (int i = 0; i < state.vision.size(); ++i) {
        // Immediate collision check (current cell)
        if (i == 0) {
            if (state.vision[i] == WALL) {
                reward += WALL_PENALTY * 2.0f;  // Double penalty for direct collision
                wall_in_path = true;
            }
            else if (state.vision[i] == FOOD) {
                reward += FOOD_REWARD * 1.5f;    // Bonus for immediate food
                food_in_path = true;
            }
            continue;
        }

        // Distance-weighted observations (closer = stronger signal)
        float distance_weight = 1.0f / (i + 1);
        
        if (state.vision[i] == WALL) {
            // Only penalize if no food exists in same direction
            if (!food_in_path) {
                reward += WALL_PENALTY * distance_weight * 0.8f;
                wall_in_path = true;
            }
        }
        else if (state.vision[i] == FOOD) {
            reward += FOOD_REWARD * distance_weight;
            food_in_path = true;
            
            // Risk-taking bonus: approaching wall-containing path with food
            if (wall_in_path) {
                reward += 0.3f * distance_weight;  // Encourage calculated risks
            }
        }
    }

    // Energy-based survival incentives
    if (state.energy_lvl <= 0) {
        reward -= 1.0f;  // Starvation penalty
    } else {
        reward += 0.15f * (state.energy_lvl / MAX_ENERGY);  // Energy conservation
    }

    return std::clamp(reward, -1.0, 1.0);
}

double computeReward(State state, Action action, std::vector<double> food_rates, uint32_t organism_sector, bool enable_rnd) {

    if (!enable_rnd) {
        // If RND is not enabled, use the extrinsic reward only
        return computeExtrinsicReward(state, action);
    }
    double* input_data = prepareInputData(state, true, food_rates, organism_sector);


    // IDs for predictor and target will both be 0 for now
    uint32_t id = 0;
    uint32_t nn_type = 2; // RND predictor

    double* predictor_q_values = new double[4]; // Assuming 4 actions
    predict_nn(id, 2, input_data, predictor_q_values); 

    nn_type = 3; // RND target
    double* target_q_values = new double[4]; // Assuming 4 actions
    predict_nn(id, 3, input_data, target_q_values);

    double intrinsic_reward = 0.0f;

    // Assuming predictor_q_values and target_q_values are 1D arrays with 4 elements each
    // Calculate the RND reward as the squared difference between predictor and target Q-values
    for (int i = 0; i < 4; ++i) {
        intrinsic_reward += (predictor_q_values[i] - target_q_values[i]) * (predictor_q_values[i] - target_q_values[i]);
    }

    

    if (!std::isfinite(intrinsic_reward)) intrinsic_reward = 1e12;
    intrinsic_reward = std::min(intrinsic_reward, 1e12);
    intrinsic_reward = std::log1p(intrinsic_reward);


    double z = stats::peek_z_score(intrinsic_reward);

    stats::update_stats(intrinsic_reward);
    
    double extrinsic_reward = computeExtrinsicReward(state, action);

    double beta = stats::current_beta();

    double total_reward = extrinsic_reward + beta * z;

    double max_q_value = target_q_values[0];

    int best_action_index = 0;
    for (int i = 1; i < 4; ++i) {
        if (target_q_values[i] > max_q_value) {
            max_q_value = target_q_values[i];
            best_action_index = i;
        }
    }

    double target = total_reward + 0.9 * max_q_value;

    target_q_values[best_action_index] = target;


    train_nn(0, 2, target_q_values);

    //delete[] input_data;
    //delete[] predictor_q_values;
    //delete[] target_q_values;



    return total_reward;


}

double* prepareInputData(State state, bool is_RND, std::vector<double> food_rates, uint32_t organism_sector) {

    if (is_RND) {
        size_t input_size = 9 + 1 + 1; // 9 represents food eating rates, 1 for energy level total, sector_organism = 11 inputs
        double* input_data = new double[input_size];  // Allocate as a single array

        input_data[0] = organism_sector;

        input_data[1] = static_cast<double>(state.energy_lvl);

        // Add food eating rates
        for (size_t i = 0; i < food_rates.size(); ++i) {
            input_data[2 + i] = food_rates[i]; // Assuming food_rates is a vector of size 9
        }


        return input_data;  // Return the prepared input data

    }

    size_t input_size = 4 + 1 + MAX_ORGANISM_VISION_DEPTH * 4; // 4 inputs represent one cell, one hot encode each cell
    double* input_data = new double[input_size];  // Allocate as a single array

    // Populate data
    input_data[0] = static_cast<double>(state.genome.gender);
    input_data[1] = static_cast<double>(state.genome.vision_depth);
    input_data[2] = static_cast<double>(state.genome.speed);
    input_data[3] = static_cast<double>(state.genome.size);
    input_data[4] = static_cast<double>(state.energy_lvl);

    // Add vision data
    for (size_t i = 0; i < state.vision.size(); i += 4) {
        if (state.vision[i] == EMPTY) {
            input_data[5 + i] = 1.0; // One-hot encoding for EMPTY
            input_data[5 + i + 1] = 0.0;
            input_data[5 + i + 2] = 0.0;
            input_data[5 + i + 3] = 0.0;
        } else if (state.vision[i] == WALL) {
            input_data[5 + i] = 0.0;
            input_data[5 + i + 1] = 1.0; // One-hot encoding for WALL
            input_data[5 + i + 2] = 0.0;
            input_data[5 + i + 3] = 0.0;
        } else if (state.vision[i] == FOOD) {
            input_data[5 + i] = 0.0;
            input_data[5 + i + 1] = 0.0;
            input_data[5 + i + 2] = 1.0; // One-hot encoding for FOOD
            input_data[5 + i + 3] = 0.0;
        } else if (state.vision[i] == ORGANISM) {
            input_data[5 + i] = 0.0;
            input_data[5 + i + 1] = 0.0;
            input_data[5 + i + 2] = 0.0;
            input_data[5 + i + 3] = 1.0; // One-hot encoding for ORGANISM
        }
    }

    return input_data;
    
}