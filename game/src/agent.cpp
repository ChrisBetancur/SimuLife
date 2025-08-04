#include <agent.h>
#include <nn_api.h>
#include <filesystem>
#include <stats.h>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <logger.h>


#define RND_DIRECTORY "rnd_models"




Agent::Agent(Organism* organism):
    m_organism(organism) {  // Dynamically allocate

    m_epsilon_policy = nullptr;
    m_boltzmann_policy = nullptr;
    //m_policy = new EpsilonGreedyPolicy(0.1, 0.9995, 0.06); // Initialize policy
    //m_policy = new BoltzmannPolicy(1.0, 0.9995, 0.1); // Initialize Boltzmann policy
}

void Agent::setPolicy(PolicyType policy_type) {
    m_policy_type = policy_type;
    // Free the existing policy if it exists
    if (m_epsilon_policy) {
        delete m_epsilon_policy;
        m_epsilon_policy = nullptr;
    }
    if (m_boltzmann_policy) {
        delete m_boltzmann_policy;
        m_boltzmann_policy = nullptr;
    }

    if (policy_type == PolicyType::EPSILON_GREEDY) {
        m_epsilon_policy = new EpsilonGreedyPolicy(0.1, 0.9995, 0.06);
    } else if (policy_type == PolicyType::BOLTZMANN) {
        m_boltzmann_policy = new BoltzmannPolicy(1.0, 0.9995, 0.1);
    } else {
        throw std::invalid_argument("Unknown policy type");
    }
}

Agent::~Agent() {  // Destructor to free memory
    if (m_epsilon_policy) {
        delete m_epsilon_policy;
    }
    if (m_boltzmann_policy) {
        delete m_boltzmann_policy;
    }
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
    switch (m_policy_type) {
        case PolicyType::EPSILON_GREEDY:
            return m_epsilon_policy->selectAction(0, 0, m_state);
        case PolicyType::BOLTZMANN:
            return m_boltzmann_policy->selectAction(0, 0, m_state);
        default:
            throw std::invalid_argument("Unknown policy type");
    }
}


Trainer::Trainer(Agent* agent, Map* map, double discount_factor, double learning_rate, std::string model_path):
    m_agent(agent),
    m_map(map),
    m_discount_factor(discount_factor),
    m_learning_rate(learning_rate) {
    // if model path does not exist, create directory and init nn
    if (!std::filesystem::exists(model_path) || model_path == "") {
        std::filesystem::create_directories(model_path);
        std::cout << "Creating new model directory: " << model_path << std::endl;
        // Initialize online neural network with default parameters
        init_nn(7, 4, 20, 4, 1, 0); // 4 for genome, 1 for energy level, 2 for vision (food count and is_wall)
    }
    else {
        const char* model_path_cstr = model_path.c_str();

        uint32_t id = load_nn_model(model_path_cstr, 0);
    }
    // Initialize target neural network
    init_nn(7, 4, 20, 4, 1, 1);
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
        init_nn(11, 4, 20, 4, 1, 2);
    }
    else {
        std::cout << "Loading RND predictor neural network from: " << full_predictor_path << std::endl;
        const char* model_path_cstr = full_predictor_path.c_str();
        uint32_t id = load_nn_model(model_path_cstr, 2);
    }

    if (!std::filesystem::exists(full_target_path)) {

        // Initialize RND target neural network
        init_nn(11, 4, 20, 4, 1, 3);
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

void Trainer::learn(State state, Action action, double reward) {


    target_nn_update_counter++;
    if (target_nn_update_counter % 100 == 0) {
        // copy the online nn to the target nn
        update_target_nn(0, 0);
    }

    // Prepare input data for the neural network
    double* input_data = prepareInputData(state, false, {}, 0);

    double* q_values = new double[4]; // Assuming 4 actions

    predict_nn(0, 1, input_data, q_values); // batch size should be 1 therefore we only expect 1 sample output

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

    Logger::getInstance().log(LogType::DEBUG, "Learning Stage Q-Values: " + 
        std::to_string(q_values[0]) + ", " + 
        std::to_string(q_values[1]) + ", " + 
        std::to_string(q_values[2]) + ", " + 
        std::to_string(q_values[3]));

    Logger::getInstance().log(LogType::DEBUG, "Best Action Index: " + std::to_string(best_action_index) +
        ", Max Q-Value: " + std::to_string(max_q_value));

    // multiply q value with discount
    double target = reward + m_discount_factor * max_q_value;

    q_values[best_action_index] = target;

    // Train the neural network
    train_nn(0, 0, q_values);

    //delete[] input_data;
    //delete[] q_values;
}


