#include <learning_rate_scheduler.h>

#include <cmath>
#include <iostream>

LearningRateScheduler::LearningRateScheduler(double initial_lr, double min_lr, int max_steps)
    : m_learning_rate(initial_lr), m_min_lr(min_lr), m_max_steps(max_steps) {}

double LearningRateScheduler::get_learning_rate(int step) {
    if (m_max_steps == 0) return m_learning_rate;

    // 1) Warmup for ~1% of steps
    int warmup = std::max(1, m_max_steps / 100); // e.g., 10k if max_steps=1e6
    if (step < warmup) {
        double w = static_cast<double>(step) / warmup;
        return m_min_lr + (m_learning_rate - m_min_lr) * w; // linear warmup
    }

    // 2) Cosine decay with gentle restarts every ~200k steps (optional)
    int cycle = 200000; // tune per project; set 0 to disable restarts
    int t = cycle ? (step - warmup) % cycle : (step - warmup);
    int T = cycle ? cycle : (m_max_steps - warmup);

    double cosine = 0.5 * (1.0 + std::cos(M_PI * static_cast<double>(t) / T));
    double lr = m_min_lr + (m_learning_rate - m_min_lr) * cosine;
    return std::max(lr, m_min_lr);
}