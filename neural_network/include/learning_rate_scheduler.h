#ifndef LEARNING_RATE_SCHEDULER_H
#define LEARNING_RATE_SCHEDULER_H

class LearningRateScheduler {

private:    
    double m_learning_rate;
    double m_min_lr;
    int m_max_steps;
public:
    // Cosine Annealing constructor
    LearningRateScheduler(double initial_lr, double min_lr, int max_epochs);

    double get_learning_rate(int step);

};

#endif