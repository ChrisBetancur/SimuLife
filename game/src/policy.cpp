#include <policy.h>
#include <logger.h>
#include <sstream>

// Policy will help agent decide what action to take
EpsilonGreedyPolicy::EpsilonGreedyPolicy(double epsilon, double decay_rate, double min_epsilon) : 
    m_epsilon(epsilon),
    m_decay_rate(decay_rate),
    m_min_epsilon(min_epsilon),
    rng(rd()),
    unif(0.0, 1.0) {
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

        double* q_values = new double[4];

        predict_nn(id, nn_type, input_data, q_values, 1); // batch size should be 1 therefore we only expect 1 sample output

        double max_q_value = q_values[0];
        int best_action_index = 0;
        for (int i = 1; i < 4; ++i) {
            if (q_values[i] > max_q_value) {
                max_q_value = q_values[i];
                best_action_index = i;
            }
        }


        Action action;
        action.direction = static_cast<Direction>(best_action_index);


        m_epsilon *= m_decay_rate;

        if (m_epsilon < m_min_epsilon) {
            m_epsilon = m_min_epsilon;
        }

        //delete[] q_values;

        return action;
    }
    
}

std::vector<double> BoltzmannPolicy::computeProbabilities(double* q_values) {
    const int num_actions = 4;
    std::vector<double> probabilities;
    probabilities.reserve(num_actions);

    // 1. Find maximum Q-value for numerical stability
    double max_q = q_values[0];
    for (int i = 1; i < num_actions; ++i) {
        if (q_values[i] > max_q) {
            max_q = q_values[i];
        }
    }

    // 2. Compute exponentials and their sum
    double sum_exp = 0.0;
    for (int i = 0; i < num_actions; ++i) {
        // Apply temperature scaling and exponentiate
        double exp_val = std::exp((q_values[i] - max_q) / m_temperature);
        probabilities.push_back(exp_val);
        sum_exp += exp_val;
    }

    // 3. Normalize to get probabilities
    for (double& prob : probabilities) {
        prob /= sum_exp;
    }

    return probabilities;
}

    // Select action based on softmax probabilities
int BoltzmannPolicy::selectAction(double* q_values) {
    std::vector<double> probs = computeProbabilities(q_values);
        
    // Create cumulative distribution
    std::vector<double> cumulative;
    cumulative.reserve(probs.size());
    cumulative.push_back(probs[0]);
        
    for (size_t i = 1; i < probs.size(); ++i) {
        cumulative.push_back(cumulative.back() + probs[i]);
    }
        
    // Sample from distribution
    double r = uniform_dist(rng);
    for (size_t i = 0; i < cumulative.size(); ++i) {
        if (r <= cumulative[i]) {
            return static_cast<int>(i);
        }
    }

    std::ostringstream ss;
    ss << "Boltzmann Policy Probs: [";
    for (size_t i = 0; i < probs.size(); ++i) {
        ss << probs[i] << (i+1<probs.size() ? ", " : "");
    }
    ss << "]"<<std::endl;
    Logger::getInstance().log(LogType::DEBUG, ss.str());
        
    return 0;  // Fallback
}
// In your agent code:
Action BoltzmannPolicy::selectAction(uint32_t id, uint32_t nn_type, State state) {
    // verify that state is not null or inv
    // Prepare input data
    double* input_data = prepareInputData(state, false, {}, 0);
    
    double* q_values = new double[4]; // Assuming 4 actions
    // Get Q-values from neural network
    predict_nn(id, nn_type, input_data, q_values, 1);
    //delete[] input_data;



    Logger::getInstance().log(LogType::DEBUG, "Boltzmann Policy Q-Values: " + 
        std::to_string(q_values[0]) + ", " + 
        std::to_string(q_values[1]) + ", " + 
        std::to_string(q_values[2]) + ", " + 
        std::to_string(q_values[3]));
    
    // Create action object
    Action action;
    action.direction = static_cast<Direction>(this->selectAction(q_values));
    
    // Update temperature
    decayTemperature();

    Logger::getInstance().log(LogType::DEBUG, "Selected Action: " + 
        std::to_string(static_cast<int>(action.direction)) + 
        " with Temperature: " + std::to_string(m_temperature));

    return action;
}

// Update temperature (call after each action selection)
void BoltzmannPolicy::decayTemperature() {
    if (++ m_decay_counter % m_decay_interval == 0) {
        m_temperature = std::max(m_min_temperature, m_temperature * m_decay_rate);
    }
    
}

// Get current temperature
double BoltzmannPolicy::getTemperature() {
    return m_temperature;
}
