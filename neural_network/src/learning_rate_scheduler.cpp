#include <learning_rate_scheduler.h>

#include <cmath>
#include <iostream>

LearningRateScheduler::LearningRateScheduler(double initial_lr, double min_lr, int max_steps)
    : m_learning_rate(initial_lr), m_min_lr(min_lr), m_max_steps(max_steps) {}

double LearningRateScheduler::get_learning_rate(int step) {
    if (m_max_steps == 0) return m_learning_rate; // Avoid division by zero
    double normalized_step = static_cast<double>(step) / m_max_steps;
    double cosine_decay = 0.5 * (1 + std::cos(M_PI * normalized_step));
    return m_min_lr + (m_learning_rate - m_min_lr) * cosine_decay;
}