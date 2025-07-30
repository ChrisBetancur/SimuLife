#ifndef AGENT_H
#define AGENT_H 

#include <organism.h>
#include <map>
#include <sprites.h>
#include <map.h>
#include <random>
#include <policy.h>
#include <rl_utils.h>

#define TARGET_NN_UPDATE_INTERVAL 1000

/*
struct State {
    Genome genome;
    double energy_lvl;
    std::vector<CellType> vision;
};

struct Action {
    Direction direction;
};*/

// daytime reward computation
double computeReward(State state, Action action, std::vector<double> food_rates = std::vector<double>(), uint32_t organism_sector = 0);

double* prepareInputData(State state, bool is_RND = false, std::vector<double> food_rates = std::vector<double>(), uint32_t organism_sector = 0);

/*
class EpsilonGreedyPolicy {
    private:
        double m_epsilon;
        double m_decay_rate;
        double m_min_epsilon;

        std::random_device rd;
        std::mt19937 rng;
        std::uniform_real_distribution<double> unif;

    public:
        EpsilonGreedyPolicy(double epsilon = 1.0, double decay_rate = 0.99, double min_epsilon = 0.01);
        
        Action selectAction(uint32_t id, uint32_t nn_type, State state);
};

class BoltzmannPolicy {
private:
    double m_temperature;      // Exploration parameter (tau)
    double m_decay_rate;       // Temperature decay rate
    double m_min_temperature;  // Minimum temperature value
    std::random_device rd;
    std::mt19937 rng;
    std::uniform_real_distribution<double> uniform_dist;

    int selectAction(const std::vector<double>& q_values);

public:
    BoltzmannPolicy(double initial_temp = 1.0, 
                    double decay_rate = 0.9995,
                    double min_temp = 0.1)
        : m_temperature(initial_temp),
          m_decay_rate(decay_rate),
          m_min_temperature(min_temp),
          rng(rd()),
          uniform_dist(0.0, 1.0) {}

    std::vector<double> computeProbabilities(double* q_values);

    int selectAction(double* q_values);

    Action selectAction(uint32_t id, uint32_t nn_type, State state);

    void decayTemperature();

    double getTemperature();
};*/

class Agent {
    private:
        Organism* m_organism;
        State m_state;
        Action m_action;

        //EpsilonGreedyPolicy* m_policy;
        BoltzmannPolicy* m_policy;

    public:
        Agent(Organism* organism);

        ~Agent();
        
        void updateState(Map* map);
        
        Action chooseAction();

        State getState() const { return m_state; }
        
        void learn(State state, Action action, float reward);
};

class Trainer {
    private:
        Agent* m_agent;
        Map* m_map;

        double m_discount_factor;
        double m_learning_rate;

        /*std::map<State, std::map<Action, double>> q_table;
        std::vector<State> replay_buffer;
        int replay_buffer_size = 1000;
        int replay_buffer_index = 0;
        int batch_size = 32;*/

        int target_nn_update_counter = 0;

    public:
        Trainer(Agent* agent, Map* map, double discount_factor = 0.9, double learning_rate = 0.001, std::string model_path = "");
        
        void updateState();
        
        Action chooseAction();
        
        void learn(State state, Action action, float reward);
};



#endif