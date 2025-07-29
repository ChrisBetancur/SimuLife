#include <agent.h>
#include <nn_api.h>
#include <filesystem>
#include <stats.h>
#include <iostream>
#include <iomanip>
#include <algorithm>   // for std::clamp


#define RND_DIRECTORY "rnd_models"

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

double computeReward(State state, Action action, std::vector<double> food_rates, uint32_t organism_sector) {


    double* input_data = prepareInputData(state, true, food_rates, organism_sector);

    // print input data
    std::cout << "->Input data: ";
    for (int i = 0; i < 11; ++i) {
        std::cout << input_data[i] << " ";
    }
    std::cout << std::endl;

    
        // Initialize RND predictor neural network
        //init_nn(25, 4, 20, 4, 1, 2);
    
        // Initialize RND target neural network
        //init_nn(25, 4, 20, 4, 1, 3);


    

    // IDs for predictor and target will both be 0 for now
    uint32_t id = 0;
    uint32_t nn_type = 2; // RND predictor
    double* predictor_q_values = predict_nn(id, 2, input_data); 

    // print predictor q values
    std::cout << "Predictor Q-values: ";
    for (int i = 0; i < 4; ++i) {
        std::cout << predictor_q_values[i] << " ";
    }
    std::cout << std::endl;

    nn_type = 3; // RND target
    double* target_q_values = predict_nn(id, 3, input_data);

    double intrinsic_reward = 0.0f;

    // Assuming predictor_q_values and target_q_values are 1D arrays with 4 elements each
    // Calculate the RND reward as the squared difference between predictor and target Q-values
    for (int i = 0; i < 4; ++i) {
        intrinsic_reward += (predictor_q_values[i] - target_q_values[i]) * (predictor_q_values[i] - target_q_values[i]);
    }

    // print target q values
    std::cout << "Target Q-values: ";
    for (int i = 0; i < 4; ++i) {
        std::cout << target_q_values[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "Intrinsic Reward: " << intrinsic_reward << std::endl;

    if (!std::isfinite(intrinsic_reward)) intrinsic_reward = 1e12;
    intrinsic_reward = std::min(intrinsic_reward, 1e12);
    intrinsic_reward = std::log1p(intrinsic_reward);

    // print intrinsic reward
    std::cout << "After clamp Intrinsic Reward: " << intrinsic_reward << std::endl;


    double z = stats::peek_z_score(intrinsic_reward);

    stats::update_stats(intrinsic_reward);

    std::cout << std::scientific
            << std::setprecision(8)
            << "Normalized Intrinsic Reward: " << z << "\n";

    
    double extrinsic_reward = computeExtrinsicReward(state, action);

    std::cout << "Extrinsic Reward: " << extrinsic_reward << std::endl;

        double beta = stats::current_beta();
    std::cout << "Beta: " << beta << std::endl;

    double total_reward = extrinsic_reward + beta * z;

    std::cout << "Total Reward: " << total_reward << std::endl;

    
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


// Policy will help agent decide what action to take
EpsilonGreedyPolicy::EpsilonGreedyPolicy(double epsilon, double decay_rate, double min_epsilon) : 
    m_epsilon(epsilon),
    m_decay_rate(decay_rate),
    m_min_epsilon(min_epsilon),
    rng(rd()),
    unif(0.0, 1.0) {
}

double* prepareInputData(State state, bool is_RND, std::vector<double> food_rates, uint32_t organism_sector) {

    if (is_RND) {
        size_t input_size = 9 + 1 + 1; // 9 represents food eating rates, 1 for energy level total, sector_organism = 11 inputs
        double* input_data = new double[input_size];  // Allocate as a single array

        input_data[0] = organism_sector;

        input_data[2] = static_cast<double>(state.energy_lvl);

        // Add food eating rates
        for (size_t i = 0; i < food_rates.size(); ++i) {
            input_data[3 + i] = food_rates[i]; // Assuming food_rates is a vector of size 9
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
        
Action EpsilonGreedyPolicy::selectAction(uint32_t id, uint32_t nn_type, State state) {
    double n = unif(rng);

    std::vector<Action> actions;

    if (n < m_epsilon) {
        // print check
        // Explore: choose a random action
        std::uniform_int_distribution<int> dist(0, 3);
        Action action;
        action.direction = static_cast<Direction>(dist(rng));
        return action;
    } 
    else {


        // Prepare input data for the neural network
        double* input_data = prepareInputData(state, false, {}, 0);

        double* q_values = predict_nn(id, nn_type, input_data); // batch size should be 1 therefore we only expect 1 sample output

        // Assuming q_values is a 2D array with shape (1, 4) for 4 actions
        // Find the action with the maximum Q-value
        double max_q_value = q_values[0];
        int best_action_index = 0;
        for (int i = 1; i < 4; ++i) {
            if (q_values[i] > max_q_value) {
                max_q_value = q_values[i];
                best_action_index = i;
            }
        }

        // print q values
        std::cout << "Q-values: ";
        for (int i = 0; i < 4; ++i) {
            std::cout << q_values[i] << " ";
        }
        std::cout << std::endl;

        Action action;
        action.direction = static_cast<Direction>(best_action_index);

        // print direction
        std::cout << "Action: " << action.direction << std::endl;

        m_epsilon *= m_decay_rate;

        if (m_epsilon < m_min_epsilon) {
            m_epsilon = m_min_epsilon;
        }

        return action;
    }
    
}

// 1 year, $320 -75 rebate, free shipping total $245
// 6 month, $170 -30 rebate, 11.99 shipping total $152


Agent::Agent(Organism* organism):
    m_organism(organism) {  // Dynamically allocate
    m_policy = new EpsilonGreedyPolicy(0.1, 0.9995, 0.06); // Initialize policy
}

Agent::~Agent() {  // Destructor to free memory
    delete m_policy; // Free the policy
}
     
void Agent::updateState(Map* map) {
    // Update the state of the agent based on the organism's properties
    m_state.genome = m_organism->getGenome();
    m_state.energy_lvl = m_organism->getEnergy();

    int x, y;
    m_organism->getPosition(x, y);
    // Get the vision of the organism
    m_state.vision = map->getVision(x, y, m_organism->getDirection(), m_organism->getGenome().vision_depth, m_organism->getGenome().size);


}

Action Agent::chooseAction() {
    // Choose an action based on the policy
    Action action = m_policy->selectAction(0, 0, m_state);
    return action;
}

void Agent::learn(State state, Action action, float reward) {
    
}

Trainer::Trainer(Agent* agent, Map* map, double discount_factor, double learning_rate, std::string model_path):
    m_agent(agent),
    m_map(map),
    m_discount_factor(discount_factor),
    m_learning_rate(learning_rate) {
    // if model path does not exist, create directory and init nn
    if (!std::filesystem::exists(model_path) || model_path == "") {
        std::filesystem::create_directories(model_path);
        // Initialize online neural network with default parameters
        init_nn(25, 4, 20, 4, 1, 0); // 20 neurons for the vision alone since we have 5 cells in the vision and each cell has 4 possible states
    }
    else {
        const char* model_path_cstr = model_path.c_str();

        uint32_t id = load_nn_model(model_path_cstr, 0);
    }
    // Initialize target neural network
    init_nn(25, 4, 20, 4, 1, 1);
    // copy the online nn to the target nn
    update_target_nn(0, 0);

    std::string full_target_path = std::string(RND_DIRECTORY)
                             + "/" 
                             + model_path 
                             + "/target";

    std::string full_predictor_path = std::string(RND_DIRECTORY)
                             + "/" 
                             + model_path 
                             + "/predictor";


    if (!std::filesystem::exists(full_predictor_path)) {
        //print check
        std::cout << "Initializing RND predictor neural network" << std::endl;
        // Initialize RND predictor neural network
        init_nn(25, 4, 20, 4, 1, 2);
    }
    else {
        std::cout << "Loading RND predictor neural network from: " << full_predictor_path << std::endl;
        const char* model_path_cstr = full_predictor_path.c_str();
        uint32_t id = load_nn_model(model_path_cstr, 2);
    }

    if (!std::filesystem::exists(full_target_path)) {

        // Initialize RND target neural network
        init_nn(25, 4, 20, 4, 1, 3);
        randomize_weights(0, 3); // Randomize weights for the target network
    }
    else {
        const char* model_path_cstr = full_target_path.c_str();
        // print check
        std::cout << "Loading RND target neural network from: " << model_path_cstr << std::endl;
        uint32_t id = load_nn_model(model_path_cstr, 3);
    }

    //exit(1);
}

void Trainer::learn(State state, Action action, float reward) {


    target_nn_update_counter++;
    if (target_nn_update_counter % 100 == 0) {
        // copy the online nn to the target nn
        update_target_nn(0, 0);
    }

    // Prepare input data for the neural network
    double* input_data = prepareInputData(state, false, {}, 0);

    double* q_values = predict_nn(0, 1, input_data); // batch size should be 1 therefore we only expect 1 sample output

    // Assuming q_values is a 2D array with shape (1, 4) for 4 actions
    // Find the action with the maximum Q-value
    double max_q_value = q_values[0];
    int best_action_index = 0;
    for (int i = 1; i < 4; ++i) {
        if (q_values[i] > max_q_value) {
            max_q_value = q_values[i];
            best_action_index = i;
        }
    }

    // multiply q value with discount
    double target = reward + m_discount_factor * max_q_value;

    q_values[best_action_index] = target;

    // Train the neural network
    train_nn(0, 0, q_values);
}


