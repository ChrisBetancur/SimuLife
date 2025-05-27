#include <agent.h>
#include <nn_api.h>
#include <filesystem>


float computeReward(State state, Action action) {
    const float WALL_PENALTY = -1.0f;    // Base penalty for wall collision
    const float FOOD_REWARD = 1.0f;      // Base reward for food
    float reward = 0.0f;
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

    return std::clamp(reward, -1.0f, 1.0f);
}


// Policy will help agent decide what action to take
EpsilonGreedyPolicy::EpsilonGreedyPolicy(double epsilon, double decay_rate, double min_epsilon) : 
    m_epsilon(epsilon),
    m_decay_rate(decay_rate),
    m_min_epsilon(min_epsilon),
    rng(rd()),
    unif(0.0, 1.0) {
}

double* prepareInputData(State state) {
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
        double* input_data = prepareInputData(state);

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
    // Initialize Q-table and replay buffer
    q_table = std::map<State, std::map<Action, double>>();
    replay_buffer = std::vector<State>();
    replay_buffer.reserve(replay_buffer_size);

    // if model path does not exist, create directory and init nn
    if (!std::filesystem::exists(model_path) || model_path == "") {
        std::filesystem::create_directories(model_path);
        init_nn(25, 4, 20, 4, 1, 0); // 20 neurons for the vision alone since we have 5 cells in the vision and each cell has 4 possible states
    }
    else {
        const char* model_path_cstr = model_path.c_str();

        uint32_t id = load_nn_model(model_path_cstr, 0);
    }
    // Initialize neural network
    
    init_nn(25, 4, 20, 4, 1, 1);
    // copy the online nn to the target nn
    update_target_nn(0, 0);
}

void Trainer::learn(State state, Action action, float reward) {


    target_nn_update_counter++;
    if (target_nn_update_counter % 100 == 0) {
        // copy the online nn to the target nn
        update_target_nn(0, 0);
    }

    // Prepare input data for the neural network
    double* input_data = prepareInputData(state);

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


