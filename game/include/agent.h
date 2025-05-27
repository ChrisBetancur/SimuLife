#ifndef AGENT_H
#define AGENT_H 

#include <organism.h>
#include <map>
#include <sprites.h>
#include <map.h>
#include <random>

#define MAX_ENERGY 100.0f
#define TARGET_NN_UPDATE_INTERVAL 1000

struct State {
    Genome genome;
    double energy_lvl;
    std::vector<CellType> vision;
};

struct Action {
    Direction direction;
};

// daytime reward computation
float computeReward(State state, Action action);

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

class Agent {
    private:
        Organism* m_organism;
        State m_state;
        Action m_action;

        EpsilonGreedyPolicy* m_policy;

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

        std::map<State, std::map<Action, double>> q_table;
        std::vector<State> replay_buffer;
        int replay_buffer_size = 1000;
        int replay_buffer_index = 0;
        int batch_size = 32;

        int target_nn_update_counter = 0;

    public:
        Trainer(Agent* agent, Map* map, double discount_factor = 0.9, double learning_rate = 0.001, std::string model_path = "");
        
        void updateState();
        
        Action chooseAction();
        
        void learn(State state, Action action, float reward);
};



#endif