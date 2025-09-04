#include <agent.h>
#include <nn_api.h>
#include <filesystem>
#include <stats.h>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <logger.h>

#define RND_DIRECTORY "rnd_models"

IO_FRONTEND::RND_Params rnd_parameters;
IO_FRONTEND::DQN_Params dqn_parameters;
IO_FRONTEND::BoltzmannPolicy_Params boltzmann_parameters;

Agent::Agent(Organism* organism):
    m_organism(organism) {  // Dynamically allocate

    //m_epsilon_policy = nullptr;
    m_boltzmann_policy = nullptr;
}

void Agent::setPolicy(PolicyType policy_type) {
    /*int status = parse_rnd_params("../game/rl_system.params", rnd_parameters);
    
    if (status == false) {
        std::cerr << "Error parsing RND parameters for frontend" << std::endl;
        exit(1);
    }

    status = parse_dqn_params("../game/rl_system.params", dqn_parameters);

    if (status == false) {
        std::cerr << "Error parsing DQN parameters for frontend" << std::endl;
        exit(1);
    }*/

    int status = parse_boltzmann_params("../game/rl_system.params", boltzmann_parameters);

    if (status == false) {
        std::cerr << "Error parsing Boltzmann parameters for frontend" << std::endl;
        exit(1);
    }

    m_policy_type = policy_type;
    // Free the existing policy if it exists
    /*if (m_epsilon_policy) {
        delete m_epsilon_policy;
        m_epsilon_policy = nullptr;
    }*/
    if (m_boltzmann_policy) {
        delete m_boltzmann_policy;
        m_boltzmann_policy = nullptr;
    }

    if (policy_type == PolicyType::EPSILON_GREEDY) {
        //m_epsilon_policy = new EpsilonGreedyPolicy(0.1, 0.9995, 0.06);
        std::cout << "IDEK" << std::endl;
        exit(1);
    } else if (policy_type == PolicyType::BOLTZMANN) {
        //m_boltzmann_policy = new BoltzmannPolicy(1.0, 0.9999995, 0.1, 2);
        m_boltzmann_policy = new BoltzmannPolicy(boltzmann_parameters.initial_temp, boltzmann_parameters.decay_rate, boltzmann_parameters.min_temp, boltzmann_parameters.decay_interval);
    } else {
        throw std::invalid_argument("Unknown policy type");
    }
}

Agent::~Agent() {  // Destructor to free memory
    /*if (m_epsilon_policy) {
        delete m_epsilon_policy;
    }*/
    if (m_boltzmann_policy) {
        delete m_boltzmann_policy;
    }
}
     
void Agent::updateState(Map* map, bool is_eating) {
    // Update the state of the agent based on the organism's properties
    m_state.genome = m_organism->getGenome();
    m_state.energy_lvl = m_organism->getEnergy();
    m_state.is_eating = is_eating; // Set the flag for new episode

    int x, y;
    m_organism->getPosition(x, y);

    // Get the vision of the organism
    m_state.vision = map->getVision(x, y, m_organism->getDirection(), m_organism->getGenome().vision_depth, m_organism->getGenome().size);
    m_state.food_count = m_organism->foodCount();
}

Action Agent::chooseAction() {
    switch (m_policy_type) {
        case PolicyType::EPSILON_GREEDY:
            //return m_epsilon_policy->selectAction(0, 0, m_state);
        case PolicyType::BOLTZMANN:
            return m_boltzmann_policy->selectAction(0, 0, m_state);
        default:
            throw std::invalid_argument("Unknown policy type");
    }
}

RND_replay_buffer createRNDReplayBuffer(int buffer_size) {
    // 1. Parse the RND parameters here
    bool status = IO_FRONTEND::parse_rnd_params("../game/rl_system.params", rnd_parameters);
    if (!status) {
        throw std::runtime_error("Failed to parse RND parameters.");
    }
    
    // 2. Now that rnd_parameters is populated, return a new RND_replay_buffer
    // The `this->` is important to access the member variable and prevent a stack variable being created.
    return RND_replay_buffer(buffer_size, rnd_parameters);
}

Trainer::Trainer(Agent* agent, Map* map, double discount_factor, double learning_rate, std::string model_path, int buffer_size):
    m_agent(agent),
    m_map(map),
    m_discount_factor(discount_factor),
    m_learning_rate(learning_rate),
    replay_buffer_size(buffer_size),
    learning_counter(0),
    m_rnd_counter(0),
    m_rnd_replay_buffer(createRNDReplayBuffer(buffer_size)) {

    parse_nn_params(); // parse the hyperparams used in the nn backend

    int status = parse_rnd_params("../game/rl_system.params", rnd_parameters);
    
    if (status == false) {
        std::cerr << "Error parsing RND parameters for frontend" << std::endl;
        exit(1);
    }

    status = parse_dqn_params("../game/rl_system.params", dqn_parameters);
    batch_size = dqn_parameters.DQN_BATCH_SIZE;

    if (status == false) {
        std::cerr << "Error parsing DQN parameters for frontend" << std::endl;
        exit(1);
    }

    std::random_device rd;
    m_gen = std::mt19937(rd());
    // if model path does not exist, create directory and init nn
    if (!std::filesystem::exists(model_path) || model_path == "") {
        std::filesystem::create_directories(model_path);
        std::cout << "Creating new model directory: " << model_path << std::endl;
        // Initialize online neural network with default parameters
        init_nn(dqn_parameters.DQN_INPUT_DIM, dqn_parameters.DQN_OUTPUT_DIM, dqn_parameters.DQN_HIDDEN_DIM, dqn_parameters.DQN_NUM_LAYERS, dqn_parameters.DQN_BATCH_SIZE, DQN_ONLINE_ID); // 4 for genome, 1 for energy level, 2 for vision (food count and is_wall)
    }
    else {
        const char* model_path_cstr = model_path.c_str();

        uint32_t id = load_nn_model(model_path_cstr, 0);
    }
    // Initialize target neural network
    init_nn(dqn_parameters.DQN_INPUT_DIM, dqn_parameters.DQN_OUTPUT_DIM, dqn_parameters.DQN_HIDDEN_DIM, dqn_parameters.DQN_NUM_LAYERS, dqn_parameters.DQN_BATCH_SIZE, DQN_TARGET_ID);
    // copy the online nn to the target nn
    update_target_nn(0, 0); // copy the online nn to the target nn at index 0

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
        init_nn(rnd_parameters.RND_INPUT_DIM, rnd_parameters.RND_OUTPUT_DIM, rnd_parameters.RND_HIDDEN_DIM, rnd_parameters.RND_NUM_LAYERS, rnd_parameters.RND_BATCH_SIZE, RND_PREDICTOR_ID);
    }
    else {
        std::cout << "Loading RND predictor neural network from: " << full_predictor_path << std::endl;
        const char* model_path_cstr = full_predictor_path.c_str();
        uint32_t id = load_nn_model(model_path_cstr, 2);
    }

    if (!std::filesystem::exists(full_target_path)) {

        // Initialize RND target neural network
        init_nn(rnd_parameters.RND_INPUT_DIM, rnd_parameters.RND_OUTPUT_DIM, rnd_parameters.RND_HIDDEN_DIM, rnd_parameters.RND_NUM_LAYERS, rnd_parameters.RND_BATCH_SIZE, RND_TARGET_ID);
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

Trainer::~Trainer() {
    // Destructor
}

void Trainer::learn_from_batch() {

    // print batch size
    std::cout << "Batch size: " << batch_size << std::endl;
    // Serves as the inputs for the neural network training
    double* states_batch = new double[batch_size * dqn_parameters.DQN_INPUT_DIM];
    double* next_states_batch = new double[batch_size * dqn_parameters.DQN_INPUT_DIM];

    // print all parameters in dqn_parameters struct
    std::cout << "DQN Parameters:" << std::endl;
    std::cout << "DQN_INPUT_DIM: " << dqn_parameters.DQN_INPUT_DIM << std::endl;
    std::cout << "DQN_OUTPUT_DIM: " << dqn_parameters.DQN_OUTPUT_DIM << std::endl;
    std::cout << "DQN_HIDDEN_DIM: " << dqn_parameters.DQN_HIDDEN_DIM << std::endl;
    std::cout << "DQN_NUM_LAYERS: " << dqn_parameters.DQN_NUM_LAYERS << std::endl;
    std::cout << "DQN_BATCH_SIZE: " << dqn_parameters.DQN_BATCH_SIZE << std::endl;

    double* rewards_batch = new double[batch_size];
    double* dones_batch = new double[batch_size];
    double* actions_batch = new double[batch_size];
    
    std::uniform_int_distribution<> distrib(0, replay_buffer.size() - 1);

    // 2. Sample from the replay buffer and populate the batches
    for (int i = 0; i < batch_size; ++i) {
        int random_index = distrib(m_gen); // Use your member generator
        const Transition& transition = replay_buffer[random_index];

        double* input_data = prepareInputData(transition.state, false, {}, 0);
        double* next_input_data = prepareInputData(transition.next_state, false, {}, 0);

        // CORRECT WAY to copy data to the batch
        // only works for non RND learning
        // Corrected
        std::copy(input_data, input_data + dqn_parameters.DQN_INPUT_DIM, states_batch + i * dqn_parameters.DQN_INPUT_DIM);
        std::copy(next_input_data, next_input_data + dqn_parameters.DQN_INPUT_DIM, next_states_batch + i * dqn_parameters.DQN_INPUT_DIM);

        delete[] input_data;
        delete[] next_input_data;
        
        rewards_batch[i] = transition.reward;
        dones_batch[i] = transition.done ? 1.0 : 0.0;
        actions_batch[i] = static_cast<int>(transition.action.direction);
    }
    

    // 3. Prepare the target values
    double* target_values = new double[batch_size * dqn_parameters.DQN_OUTPUT_DIM];
    predict_nn(0, DQN_TARGET_ID, next_states_batch, target_values, batch_size);

    double* online_q_values = new double[batch_size * dqn_parameters.DQN_OUTPUT_DIM];
    predict_nn(0, DQN_ONLINE_ID, states_batch, online_q_values, batch_size);

    for (int i = 0; i < batch_size; ++i) {
        double max_next_q = *std::max_element(target_values + i * dqn_parameters.DQN_OUTPUT_DIM, target_values + (i + 1) * dqn_parameters.DQN_OUTPUT_DIM);
        for (int j = 0; j < dqn_parameters.DQN_OUTPUT_DIM; ++j) {
            if (j == static_cast<int>(actions_batch[i])) {
                target_values[i * dqn_parameters.DQN_OUTPUT_DIM + j] = rewards_batch[i] + (1.0 - dones_batch[i]) * m_discount_factor * max_next_q;
            } else {
                target_values[i * dqn_parameters.DQN_OUTPUT_DIM + j] = online_q_values[i * dqn_parameters.DQN_OUTPUT_DIM + j];
            }
        }
    }

    // 4. Train the online network
    train_nn(0, DQN_ONLINE_ID, states_batch, target_values, batch_size);
    

}

void Trainer::rnd_learn_from_batch() {
    if (m_rnd_replay_buffer.current_size() < rnd_parameters.RND_BATCH_SIZE) {
        return; // Not enough data to learn
    }

    double* input_data = m_rnd_replay_buffer.get_batch(rnd_parameters.RND_BATCH_SIZE);
    double* target_data = new double[rnd_parameters.RND_BATCH_SIZE * rnd_parameters.RND_OUTPUT_DIM];

    // Predict using the predictor network
    predict_nn(0, RND_PREDICTOR_ID, input_data, target_data, rnd_parameters.RND_BATCH_SIZE);

    // Train the target network
    train_nn(0, RND_TARGET_ID, input_data, target_data, rnd_parameters.RND_BATCH_SIZE);

    delete[] input_data;
    delete[] target_data;
}

void Trainer::learn(State state, State prevState, Action action, double reward, bool isDone,
                    std::vector<double> food_rates, uint32_t organism_sector) {
    // Create the full transition tuple
    Transition transition;
    transition.state = prevState;
    transition.action = action;
    transition.reward = reward;
    transition.next_state = state;
    transition.done = isDone;

    // Add the transition to the replay buffer
    updateReplayBuffer(transition);

    m_rnd_replay_buffer.add(prepareInputData(state, true, food_rates, organism_sector));

    // Update target network periodically
    target_nn_update_counter++;
    if (target_nn_update_counter % 1000 == 0) {
        update_target_nn(0, 0);
    }

    // Periodically learn from a batch
    if (learning_counter == 4) {
        if (replay_buffer.size() > batch_size) {
            learn_from_batch();
            learning_counter = 0;
        }
    } else {
        learning_counter++;
    }

    if (m_rnd_counter == 100) {
        rnd_learn_from_batch();
        m_rnd_counter = 0;
    } else {
        m_rnd_counter++;
    }
}

void Trainer::updateReplayBuffer(Transition transition) {
    if (replay_buffer.size() < replay_buffer_size) {
        replay_buffer.push_back(transition);
    } else {
        // delete the oldest state and add the new one
        replay_buffer.erase(replay_buffer.begin());
        replay_buffer.push_back(transition);
    }
}


