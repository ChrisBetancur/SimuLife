#include <rl_utils.h>
#include <stats.h>
#include <logger.h>

#include <cmath>



double computeExtrinsicReward(State state, Action action, bool hit_wall, int org_x, int org_y, 
    Direction dir, int wall_pos_x, int wall_pos_y) {
    // Extract vision information
    int foodCount;
    bool sawWall;
    int wall_distance = -1; // Default to -1 if no wall is encountered
    std::tie(foodCount, sawWall, wall_distance) = state.vision;
    
    // SCALED-DOWN extrinsic reward constants (copy/paste)
    constexpr double WALL_COLLISION_PENALTY      = -4.0;   // was -10
    constexpr double REPEATED_COLLISION_PENALTY  = -1.0;   // was -2 (smaller repeat penalty)
    constexpr double WALL_NEAR_PENALTY           = -2.0;   // was -10
    constexpr double MOVE_AWAY_WEIGHT            = 0.20;   // was 0.5 (smaller reward per distance)
    constexpr double SAFETY_BONUS                = 0.10;   // once outside NEAR_THRESHOLD (was 0.5)
    constexpr double SAFE_STEP_BONUS             = 0.02;   // small positive for safe steps
    constexpr double FOOD_REWARD                 = 0.50;   // keep modest incentive for food
    constexpr double ENERGY_WEIGHT               = 0.15;   // unchanged (ok)
    constexpr double STEP_COST                   = -0.01;  // smaller living cost (was -0.05)
    constexpr int    FOOD_COUNT_CAP              = 3;
    constexpr double NEAR_THRESHOLD              = 5.0;
    constexpr double NO_MOVE_AWAY_PENALTY        = -0.20;  // was -1.0 (much smaller)
    constexpr double MAX_DIST                    = 100.0;

    
    double reward = 0.0;

    /*static double last_distToWall;
    static int consecWallHits;

    if (state.is_new_episode) {
        last_distToWall = NEAR_THRESHOLD + 1.0;
        consecWallHits = 0;
    }*/

    // 1) starvation / energy bonus
    if (state.energy_lvl <= 0) {
        reward -= 1.0;
    } else {
        reward += ENERGY_WEIGHT * (state.energy_lvl / MAX_ENERGY);
    }

    // 2) wall interaction
    if (hit_wall) {
        reward += WALL_COLLISION_PENALTY;
    }
    else if (sawWall && wall_distance >= 0 && wall_distance < NEAR_THRESHOLD) {
        // near a wall
        reward += WALL_NEAR_PENALTY;

        // encourage moving away from the wall
        double distToWall = static_cast<double>(wall_distance);
        double distDelta = distToWall - NEAR_THRESHOLD; // positive if moving away, negative if moving closer

        if (distDelta > 0) {
            reward += MOVE_AWAY_WEIGHT * (distDelta / NEAR_THRESHOLD);
            if (distToWall > NEAR_THRESHOLD) {
                reward += SAFETY_BONUS;
            }
        } else if (distDelta < 0) {
            reward += MOVE_AWAY_WEIGHT * (distDelta / NEAR_THRESHOLD); // negative contribution
        } else {
            reward += NO_MOVE_AWAY_PENALTY;
        }

    } 
    else {
        // safe step bonus
        reward += SAFE_STEP_BONUS;
    }




    // 3) food reward (capped)
    int visibleFood = foodCount;//std::min(foodCount, FOOD_COUNT_CAP);
    reward += FOOD_REWARD * visibleFood;

    // 4) small living cost
    reward += STEP_COST;

    // clamp to [-1,1]
    //return std::clamp(reward, -1.0, +1.0);

    // scale the reward to [-15, 15] but not clamped


    //return std::clamp(reward, -15.0, 15.0);

    return reward; // try this

    //reward = std::clamp(reward, -8.0, 8.0); // delete outliers

    // scale using tanh_scale(double x, double amplitude, double sensitivity)
    //double amplitude = 6.0; // max reward
    //double sensitivity = 4.0; // how quickly the reward saturates
    //return tanh_scale(reward, amplitude, sensitivity);

}



double computeReward(State state, Action action, std::vector<double> food_rates, uint32_t organism_sector, 
    bool enable_rnd, bool hit_wall, int org_x, int org_y,
    Direction dir, int wall_pos_x, int wall_pos_y) {

    if (!enable_rnd) {
        // If RND is not enabled, use the extrinsic reward only
        double extrinsic_reward = computeExtrinsicReward(state, action, hit_wall, org_x, org_y, dir, wall_pos_x, wall_pos_y);
        Logger::getInstance().log(LogType::DEBUG, "Extrinsic Reward: " + std::to_string(extrinsic_reward));
        return extrinsic_reward;

    }
    double* input_data = prepareInputData(state, true, food_rates, organism_sector);


    // IDs for predictor and target will both be 0 for now
    uint32_t id = 0;
    uint32_t nn_type = 2; // RND predictor

    double* pred_out = new double[RND_OUTPUT_DIM];
    predict_nn(id, 2, input_data, pred_out, 1); 



    nn_type = 3; // RND target
    double* targ_out = new double[RND_OUTPUT_DIM]; // Assuming 64 outputs
    predict_nn(id, 3, input_data, targ_out, 1);


    for (int i=0;i<RND_OUTPUT_DIM;++i) {
        // print pred out[i] and targ_out[i]
        std::cout << "Output pred_out[i]=" << pred_out[i] << ", targ_out[i]=" << targ_out[i] << std::endl;
        if (!std::isfinite(pred_out[i]) || !std::isfinite(targ_out[i])) {
            std::cerr << "RND NaN at dim " << i << std::endl;
            exit(1);
        }
    }

    double intrinsic_reward = 0.0f;

    double mse = 0.0;
    double mean_abs_t = 0.0;
    for (int i = 0; i < RND_OUTPUT_DIM; ++i) {
        double d = pred_out[i] - targ_out[i];
        mse += d * d;
        mean_abs_t += std::abs(targ_out[i]);
    }
    mse /= double(RND_OUTPUT_DIM);
    double rmse = std::sqrt(mse);
    mean_abs_t /= double(RND_OUTPUT_DIM);

    // 2) relative rmse (avoid divide-by-zero)
    double rel_rmse = rmse / (1.0 + mean_abs_t);   // or use max(mean_abs_t, eps)

    // 3) compress if needed
    double metric = rel_rmse;          // baseline
    // if rel_rmse sometimes >> 1e3, compress:
    /*metric = std::log1p(rel_rmse);    // safer dynamic range

    // 4) safety checks
    if (!std::isfinite(metric)) metric = 1e12;
    metric = std::min(metric, 1e12);*/

    Logger::getInstance().log(LogType::DEBUG, "Intrinsic Reward (MSE):" + std::to_string(metric));




    //double z = stats::peek_z_score(intrinsic_reward);
    double z = stats::peek_z_score(metric);

    Logger::getInstance().log(LogType::DEBUG, "Z-Score: " + std::to_string(z));


    //stats::update_stats(intrinsic_reward);
    stats::update_stats(metric);
    
    double extrinsic_reward = computeExtrinsicReward(state, action, hit_wall, org_x, org_y, dir, wall_pos_x, wall_pos_y);

    //extrinsic_reward = std::max(-15.0, std::min(15.0, extrinsic_reward));

    Logger::getInstance().log(LogType::DEBUG, "Extrinsic Reward: " + std::to_string(extrinsic_reward));

    double beta = stats::current_beta(state.food_count);

    Logger::getInstance().log(LogType::DEBUG, "Beta: " + std::to_string(beta));

    constexpr double INTRINSIC_SCALE = 0.9;
    constexpr double INTRINSIC_CLAMP = 15.0;

    double intrinsic_term = beta * (z * INTRINSIC_SCALE);
    //intrinsic_term = std::clamp(intrinsic_term, -INTRINSIC_CLAMP, INTRINSIC_CLAMP);

    double total_reward = extrinsic_reward + intrinsic_term;

    Logger::getInstance().log(LogType::DEBUG, "Total Reward: " + std::to_string(total_reward) + " (Beta: " + std::to_string(beta) + ")");


    //train_nn(0, RND_PREDICTOR_ID, targ_out);

    // print that RND is not fully implemented yet
    std::cout << "RND intrinsic reward not fully implemented yet!" << std::endl;
    exit(1);


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