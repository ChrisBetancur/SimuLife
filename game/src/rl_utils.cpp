#include <rl_utils.h>
#include <stats.h>
#include <logger.h>

#include <cmath>

class RND_replay_buffer {
    private:
        std::vector<std::vector<double>> buffer;
        size_t capacity;
        size_t size;
        std::mt19937 m_gen;

    public:
        RND_replay_buffer(size_t capacity) : capacity(capacity), size(0) {
            buffer.reserve(capacity);
            std::random_device rd;
            m_gen = std::mt19937(rd());
        }

        void add(double* value) {
            // convert value to vector
            std::vector<double> vec(value, value + RND_INPUT_DIM);
            if (size < capacity) {
                buffer.push_back(vec);
                size++;
            } else {
                // overwrite the oldest entry
                buffer[size % capacity] = vec;
            }
        }

        double* get_batch(size_t batch_size) {
            if (batch_size > size) {
                throw std::runtime_error("Batch size exceeds current buffer size");
            }
            double* batch = new double[batch_size * RND_INPUT_DIM];
            std::uniform_int_distribution<> distrib(0, size - 1);
            for (size_t i = 0; i < batch_size; ++i) {
                int random_index = distrib(m_gen);
                std::copy(buffer[random_index].begin(), buffer[random_index].end(), batch + i * RND_INPUT_DIM);
            }

            return batch;  // Return the batch of data
        }

        size_t current_size() const {
            return size;
        }
};

RND_replay_buffer rnd_replay_buffer(1000);
int rnd_counter = 0;

double computeIntrinsicReward(int rnd_batch_size) {
    if (rnd_replay_buffer.current_size() < rnd_batch_size) {
        return 0.0; // Not enough data to compute intrinsic reward
    }
    double* input_data = rnd_replay_buffer.get_batch(rnd_batch_size);

    double* pred_out = new double[RND_OUTPUT_DIM * rnd_batch_size];
    predict_nn(0, RND_PREDICTOR_ID, input_data, pred_out, rnd_batch_size);

    double* targ_out = new double[RND_OUTPUT_DIM * rnd_batch_size];
    predict_nn(0, RND_TARGET_ID, input_data, targ_out, rnd_batch_size);

    double intrinsic_reward = 0.0f;
    double mse = 0.0;

    double mean_abs_t = 0.0;

    for (int i = 0; i < rnd_batch_size * RND_OUTPUT_DIM; ++i) {
        double d = pred_out[i] - targ_out[i];
        mse += d * d;
        mean_abs_t += std::abs(targ_out[i]);
    }

    mse /= double(rnd_batch_size * RND_OUTPUT_DIM);
    double rmse = std::sqrt(mse);
    mean_abs_t /= double(rnd_batch_size * RND_OUTPUT_DIM);
    // 2) relative rmse (avoid divide-by-zero)
    double rel_rmse = rmse / (1.0 + mean_abs_t);
    // 3) compress if needed
    double metric = rel_rmse;          // baseline


    Logger::getInstance().log(LogType::DEBUG, "Intrinsic Reward (MSE):" + std::to_string(metric));

    //double z = stats::peek_z_score(intrinsic_reward);
    double z = stats::peek_z_score(metric);

    Logger::getInstance().log(LogType::DEBUG, "Z-Score: " + std::to_string(z));


    //stats::update_stats(intrinsic_reward);
    stats::update_stats(metric);


    if (rnd_counter == 6) {
        train_nn(0, RND_PREDICTOR_ID, input_data, targ_out, 1);
        rnd_counter = 0;
    } else {
        rnd_counter++;
    }

    delete[] input_data;
    delete[] pred_out;
    delete[] targ_out;

    return z;

}

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

    //return reward; // try this

    //reward = std::clamp(reward, -8.0, 8.0); // delete outliers

    // scale using tanh_scale(double x, double amplitude, double sensitivity)
    double amplitude = 6.0; // max reward
    double sensitivity = 4.0; // how quickly the reward saturates
    return tanh_scale(reward, amplitude, sensitivity);

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
    rnd_replay_buffer.add(input_data);

    double z = computeIntrinsicReward(RND_BATCH_SIZE);
    
    double extrinsic_reward = computeExtrinsicReward(state, action, hit_wall, org_x, org_y, dir, wall_pos_x, wall_pos_y);

    //extrinsic_reward = std::max(-15.0, std::min(15.0, extrinsic_reward));

    Logger::getInstance().log(LogType::DEBUG, "Extrinsic Reward: " + std::to_string(extrinsic_reward));

    double beta = stats::current_beta(state.food_count);

    Logger::getInstance().log(LogType::DEBUG, "Beta: " + std::to_string(beta));

    constexpr double INTRINSIC_SCALE = 1.0;
    constexpr double INTRINSIC_CLAMP = 30.0;

    double intrinsic_term = beta * (z * INTRINSIC_SCALE);
    //intrinsic_term = std::clamp(intrinsic_term, -INTRINSIC_CLAMP, INTRINSIC_CLAMP);

    double total_reward = extrinsic_reward + intrinsic_term;

    Logger::getInstance().log(LogType::DEBUG, "Total Reward: " + std::to_string(total_reward) + " (Beta: " + std::to_string(beta) + ")");



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