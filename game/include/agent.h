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

class Agent {
    private:
        Organism* m_organism;
        State m_state;
        Action m_action;

        EpsilonGreedyPolicy* m_epsilon_policy;
        BoltzmannPolicy* m_boltzmann_policy;

        PolicyType m_policy_type;

    public:
        Agent(Organism* organism);

        void setPolicy(PolicyType policy_type);

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
        
        void learn(State state, Action action, double reward);
};



#endif