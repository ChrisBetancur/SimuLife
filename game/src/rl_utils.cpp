#include <rl_utils.h>
#include <stats.h>
#include <logger.h>

#include <cmath>

RND_replay_buffer::RND_replay_buffer(size_t capacity, IO_FRONTEND::RND_Params rnd_parameters) 
    : capacity(capacity), size(0), m_rnd_parameters(rnd_parameters) {
    buffer.reserve(capacity);
    std::random_device rd;
    m_gen = std::mt19937(rd());
}

// Function implementations
void RND_replay_buffer::add(double* value) {
    // convert value to vector
    std::vector<double> vec(value, value + m_rnd_parameters.RND_INPUT_DIM);
    if (size < capacity) {
        buffer.push_back(vec);
        size++;
    } else {
        // overwrite the oldest entry
        buffer[size % capacity] = vec;
    }
}

double* RND_replay_buffer::get_batch(size_t batch_size) {
    if (batch_size > size) {
        throw std::runtime_error("Batch size exceeds current buffer size");
    }
    double* batch = new double[batch_size * m_rnd_parameters.RND_INPUT_DIM];
    std::uniform_int_distribution<> distrib(0, size - 1);
    for (size_t i = 0; i < batch_size; ++i) {
        int random_index = distrib(m_gen);
        std::copy(buffer[random_index].begin(), buffer[random_index].end(), batch + i * m_rnd_parameters.RND_INPUT_DIM);
    }
    return batch;
}

size_t RND_replay_buffer::current_size() const {
    return size;
}

double computeIntrinsicReward(double* input_data) {

    IO_FRONTEND::RND_Params rnd_parameters;
    parse_rnd_params("../game/rl_system.params", rnd_parameters);

    double* pred_out = new double[rnd_parameters.RND_OUTPUT_DIM];
    predict_nn(0, RND_PREDICTOR_ID, input_data, pred_out, 1); // Pass batch size of 1

    double* targ_out = new double[rnd_parameters.RND_OUTPUT_DIM];
    predict_nn(0, RND_TARGET_ID, input_data, targ_out, 1); // Pass batch size of 1

    // 2. Compute the MSE for this single state
    double mse = 0.0;
    double mean_abs_t = 0.0;

    for (int i = 0; i < rnd_parameters.RND_OUTPUT_DIM; ++i) {
        if (std::isnan(pred_out[i]) || std::isinf(pred_out[i]) ||
            std::isnan(targ_out[i]) || std::isinf(targ_out[i])) {
            std::cerr << "Error: NaN or infinite value encountered in intrinsic reward computation." << std::endl;
            exit(1);
        }
        double d = pred_out[i] - targ_out[i];
        mse += d * d;
        mean_abs_t += std::abs(targ_out[i]);
    }
    
    mse /= double(rnd_parameters.RND_OUTPUT_DIM);
    double rmse = std::sqrt(mse);
    mean_abs_t /= double(rnd_parameters.RND_OUTPUT_DIM);

    double rel_rmse = rmse / (1.0 + mean_abs_t);
    double metric = rel_rmse;

    // 3. Update stats and get Z-score
    stats::update_stats(metric);
    double z = stats::peek_z_score(metric);

    Logger::getInstance().log(LogType::DEBUG, "Intrinsic Reward (MSE): " + std::to_string(metric));
    Logger::getInstance().log(LogType::DEBUG, "Z-Score: " + std::to_string(z));

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
    constexpr double WALL_COLLISION_PENALTY      = -15.0;   // was -10
    constexpr double FOOD_REWARD                 = 10.0;   // keep modest incentive for food
    constexpr double SEE_FOOD_REWARD             = 5.0;   // was 1.0

    double reward = 0.0;



    // 2) wall interaction
    if (hit_wall) {
        reward += WALL_COLLISION_PENALTY;
    }

    reward += FOOD_REWARD * foodCount;

    if (state.is_eating) {
        reward += FOOD_REWARD; // bonus for eating'
    }

    reward -= 0.1 * (1 - state.energy_lvl);

    //reward = std::clamp(reward, -8.0, 8.0); // delete outliers

    // scale using tanh_scale(double x, double amplitude, double sensitivity)
    /*double amplitude = 30.0; // max reward
    double sensitivity = 1.0; // how quickly the reward saturates
    return tanh_scale(reward, amplitude, sensitivity);*/

    // clamp reward to be between -15 and 15
    
    constexpr double REWARD_CAP = 20.0; 
    return std::clamp(reward, -REWARD_CAP, REWARD_CAP);

    //return reward;

}



double computeReward(State state, Action action, std::vector<double> food_rates, uint32_t organism_sector, 
    bool enable_rnd, bool hit_wall, int org_x, int org_y,
    Direction dir, int wall_pos_x, int wall_pos_y) {

    IO_FRONTEND::RND_Params rnd_parameters;
    IO_FRONTEND::parse_rnd_params("../game/rl_system.params", rnd_parameters);

    IO_FRONTEND::DQN_Params dqn_parameters;
    IO_FRONTEND::parse_dqn_params("../game/rl_system.params", dqn_parameters);

    if (!enable_rnd) {
        // If RND is not enabled, use the extrinsic reward only
        double extrinsic_reward = computeExtrinsicReward(state, action, hit_wall, org_x, org_y, dir, wall_pos_x, wall_pos_y);
        Logger::getInstance().log(LogType::DEBUG, "Extrinsic Reward: " + std::to_string(extrinsic_reward));
        return extrinsic_reward;
    }

    double* input_data = prepareInputData(state, true, food_rates, organism_sector);

    double z = computeIntrinsicReward(input_data);
    
    double extrinsic_reward = computeExtrinsicReward(state, action, hit_wall, org_x, org_y, dir, wall_pos_x, wall_pos_y);

    //extrinsic_reward = std::max(-15.0, std::min(15.0, extrinsic_reward));

    Logger::getInstance().log(LogType::DEBUG, "Extrinsic Reward: " + std::to_string(extrinsic_reward));

    double beta = stats::current_beta(state.food_count);

    Logger::getInstance().log(LogType::DEBUG, "Beta: " + std::to_string(beta));


    constexpr double INTRINSIC_SCALE = 1.0;
    constexpr double INTRINSIC_CLAMP = 30.0;

    double intrinsic_term = beta * std::max(0.0, z); // we only want positive intrinsic rewards, just because something isn't novel doesn't mean isn't good
    //intrinsic_term = std::clamp(intrinsic_term, 0.0, INTRINSIC_CLAMP);

    double total_reward = extrinsic_reward + intrinsic_term;

    /*constexpr double AMPLITUDE = 5.0; // max reward can be +/- this value
    constexpr double SENSITIVITY = 2.0; // how quickly the reward saturates
    total_reward = tanh_scale(total_reward, AMPLITUDE, SENSITIVITY);*/

    Logger::getInstance().log(LogType::DEBUG, "Total Reward: " + std::to_string(total_reward) + " (Beta: " + std::to_string(beta) + ")");

    delete[] input_data;
 
    return total_reward;
}

double* prepareInputData(State state, bool is_RND, std::vector<double> food_rates, uint32_t organism_sector) {

    IO_FRONTEND::RND_Params rnd_parameters;
    IO_FRONTEND::parse_rnd_params("../game/rl_system.params", rnd_parameters);

    IO_FRONTEND::DQN_Params dqn_parameters;
    IO_FRONTEND::parse_dqn_params("../game/rl_system.params", dqn_parameters);

    if (is_RND) {
        size_t input_size = rnd_parameters.RND_INPUT_DIM; // 9 represents food eating rates, 1 for energy level total, sector_organism = 11 inputs
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

    size_t input_size = dqn_parameters.DQN_INPUT_DIM; // 4 for genome, 1 for energy level, 2 for vision (food count and is_wall), is_eating
    double* input_data = new double[input_size];  // Allocate as a single array

    // verify that state is not null
    if (&state == nullptr) {
        std::cerr << "Error: state is null" << std::endl;
        exit(1);
    }

    // Populate data
    input_data[0] = static_cast<double>(state.genome.gender);
    input_data[1] = static_cast<double>(state.genome.vision_depth);
    input_data[2] = static_cast<double>(state.genome.speed);
    input_data[3] = static_cast<double>(state.genome.size);
    input_data[4] = static_cast<double>(state.energy_lvl);

    input_data[5] = static_cast<double>(std::get<0>(state.vision)); // food_count
    input_data[6] = static_cast<double>(std::get<1>(state.vision)); // is_wall
    input_data[7] = static_cast<double>(state.is_eating); // wall_distance


    return input_data;
    
}