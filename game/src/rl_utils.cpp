#include <rl_utils.h>
#include <stats.h>
#include <logger.h>


double computeExtrinsicReward(State state, Action action, std::vector<State> replay_buffer, bool hit_wall, int org_x, int org_y, 
    Direction dir, int wall_pos_x, int wall_pos_y) {
    // Extract vision information
    int foodCount;
    bool sawWall;
    std::tie(foodCount, sawWall) = state.vision;
    
    constexpr double WALL_COLLISION_PENALTY   = -5.0;  // collision
    constexpr double WALL_NEAR_PENALTY        = -2.0;  // proximity
    constexpr double FOOD_REWARD              = +3.0;  
    constexpr double ENERGY_WEIGHT            = 0.15;  
    constexpr double STEP_COST                = -0.05;
    constexpr int    FOOD_COUNT_CAP           = 3;
    constexpr double NEAR_THRESHOLD           = 5.0;   // e.g. pixels or units
    
    double reward = 0.0;

    // 1) starvation / energy bonus
    if (state.energy_lvl <= 0) {
        reward -= 1.0;
    } else {
        reward += ENERGY_WEIGHT * (state.energy_lvl / MAX_ENERGY);
    }

    double distToWall = std::hypot(org_x - wall_pos_x, org_y - wall_pos_y);
    double proximityFactor = std::max(0.0, (NEAR_THRESHOLD - distToWall) / NEAR_THRESHOLD);
    reward += WALL_NEAR_PENALTY * proximityFactor;

    // 2) wall collision penalty
    if (hit_wall) {
        reward += WALL_COLLISION_PENALTY;

        if (wall_pos_x != -1 && wall_pos_y != -1) {
            // If the direction of the organism is toward the wall, apply a penalty
            if ((dir == UP && org_y + 10 > wall_pos_y) ||
                (dir == DOWN && org_y - 10 < wall_pos_y) ||
                (dir == LEFT && org_x - 10 < wall_pos_x) ||
                (dir == RIGHT && org_x + 10 > wall_pos_x)) {
                reward += WALL_COLLISION_PENALTY; // double penaltly for attempting to move into a wall
                // print check
                std::cout << "Applied double wall collision penalty for attempting to move into a wall." << std::endl;
            }
        }
    }

    // based on the replay buffer, if the organism is close to a wall in any of its last 3 states, apply a small penalty
    for (int i = std::max(0, static_cast<int>(replay_buffer.size()) - 3); i < replay_buffer.size(); ++i) {
        int pastFoodCount;
        bool pastSawWall;
        std::tie(pastFoodCount, pastSawWall) = replay_buffer[i].vision;
        if (pastSawWall) {
            reward += proximityFactor * WALL_NEAR_PENALTY; // apply a small penalty for being near a wall in the past
            break; // only apply once
        }
    }

    // 3) food reward (capped)
    int visibleFood = foodCount;//std::min(foodCount, FOOD_COUNT_CAP);
    reward += FOOD_REWARD * visibleFood;

    // 4) small living cost
    reward += STEP_COST;

    // clamp to [-1,1]
    //return std::clamp(reward, -1.0, +1.0);
    return reward; // no clamping for now, let the neural network handle it
}



double computeReward(State state, Action action, std::vector<double> food_rates, uint32_t organism_sector, 
    bool enable_rnd, std::vector<State> replay_buffer, bool hit_wall, int org_x, int org_y,
    Direction dir, int wall_pos_x, int wall_pos_y) {

    if (!enable_rnd) {
        // If RND is not enabled, use the extrinsic reward only
        return computeExtrinsicReward(state, action, replay_buffer, hit_wall, org_x, org_y, dir, wall_pos_x, wall_pos_y);
    }
    double* input_data = prepareInputData(state, true, food_rates, organism_sector);


    // IDs for predictor and target will both be 0 for now
    uint32_t id = 0;
    uint32_t nn_type = 2; // RND predictor

    double* predictor_q_values = new double[4]; // Assuming 4 actions
    predict_nn(id, 2, input_data, predictor_q_values); 

    if (std::isnan(predictor_q_values[0]) || std::isnan(predictor_q_values[1]) ||
        std::isnan(predictor_q_values[2]) || std::isnan(predictor_q_values[3])) {
        std::cerr << "Error: NaN values in predictor Q-values" << std::endl;
        exit(1);
    }

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

    Logger::getInstance().log(LogType::DEBUG, "Z-Score: " + std::to_string(z));


    stats::update_stats(intrinsic_reward);
    
    double extrinsic_reward = computeExtrinsicReward(state, action, replay_buffer, hit_wall, org_x, org_y, dir, wall_pos_x, wall_pos_y);

    Logger::getInstance().log(LogType::DEBUG, "Extrinsic Reward: " + std::to_string(extrinsic_reward));

    double beta = stats::current_beta();

    double total_reward = extrinsic_reward + beta * z;

    Logger::getInstance().log(LogType::DEBUG, "Total Reward: " + std::to_string(total_reward) + " (Beta: " + std::to_string(beta) + ")");

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

        // print input data where each array slot is defined with the name


        return input_data;  // Return the prepared input data

    }

    size_t input_size = 4 + 1 + 2; // 4 for genome, 1 for energy level, 2 for vision (food count and is_wall)
    double* input_data = new double[input_size];  // Allocate as a single array

    // Populate data
    input_data[0] = static_cast<double>(state.genome.gender);
    input_data[1] = static_cast<double>(state.genome.vision_depth);
    input_data[2] = static_cast<double>(state.genome.speed);
    input_data[3] = static_cast<double>(state.genome.size);
    input_data[4] = static_cast<double>(state.energy_lvl);

    input_data[5] = static_cast<double>(std::get<0>(state.vision)); // food_count
    input_data[6] = static_cast<double>(std::get<1>(state.vision)); // is_wall
    
    // 1. i=0, 4<25, [8]
    // 2. i=4, 8<25, [12]
    // 3. i=8, 12<25, [16]
    // 4. i=12, 16<25, [20]
    // 5. i=16, 20<25, [24]
    // 6. i=20, 24<25, [28]

    // Add vision data
    /*size_t i;
    for (i = 0; i + 4 < state.vision.size(); i += 4) {
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
    }*/

    return input_data;
    
}