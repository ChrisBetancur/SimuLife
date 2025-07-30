#ifndef POLICY_H
#define POLICY_H

#include <organism.h>
#include <map>
#include <sprites.h>
#include <map.h>
#include <random>
#include <rl_utils.h>
#include <nn_api.h>

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
};

#endif // POLICY_H