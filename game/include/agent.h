#ifndef AGENT_H
#define AGENT_H 

#include <organism.h>
#include <map>
#include <sprites.h>
#include <map.h>
#include <random>
#include <policy.h>
#include <rl_utils.h>
#include <io_frontend.h>

#define TARGET_NN_UPDATE_INTERVAL 1000

extern IO_FRONTEND::RND_Params rnd_parameters;
extern IO_FRONTEND::DQN_Params dqn_parameters;
extern IO_FRONTEND::BoltzmannPolicy_Params boltzmann_parameters;

class Agent {
    private:
        Organism* m_organism;
        State m_state;
        Action m_action;

        BoltzmannPolicy* m_boltzmann_policy;

        PolicyType m_policy_type;

    public:
        Agent(Organism* organism);

        void setPolicy(PolicyType policy_type);

        ~Agent();
        
        void updateState(Map* map, bool is_eating);
        
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

        
        std::vector<Transition> replay_buffer;
        int replay_buffer_size;
        int batch_size;
        int learning_counter;


        RND_replay_buffer m_rnd_replay_buffer;
        int m_rnd_counter;

        int target_nn_update_counter;
        std::mt19937 m_gen;

        bool m_rndEnabled;

    public:
        Trainer(Agent* agent, Map* map, double discount_factor, double learning_rate, std::string model_path, int buffer_size, bool enable_rnd);

        ~Trainer();
        
        void updateState();
        
        Action chooseAction();

        void learn_from_batch();

        void rnd_learn_from_batch();
        
        void learn(State state, State prevState, Action action, double reward, bool isDone = false, 
                   std::vector<double> food_rates = {}, uint32_t organism_sector = 0);

        void updateReplayBuffer(Transition transition);

        std::vector<Transition> getReplayBuffer() const { return replay_buffer; }

        void setRNDEnabled(bool enabled) { m_rndEnabled = enabled; }


};



#endif