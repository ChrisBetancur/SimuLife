#ifndef STATS_H
#define STATS_H

#include <cstddef>  // for std::size_t
#include <cmath>    // for std::sqrt
#include <algorithm> // for std::min

namespace stats {

inline size_t n    = 0;
inline double mu   = 0.0;
inline double m2   = 0.0;

// New constant for the EMA smoothing factor
inline constexpr double EMA_ALPHA = 0.1; // A value between 0.0 and 1.0. Lower values for slower decay.

inline constexpr double beta_init = 5.0;
inline constexpr double beta_floor = 0.01;
// You'll need to tune this lambda value. A good starting point is between 2.0 and 5.0.
inline constexpr double beta_decay_lambda = 0.1;
inline constexpr std::size_t beta_decay_steps = 20000000000ULL;

inline double current_beta(int food_count) {
    double frac = std::min(1.0, double(n) / beta_decay_steps);
    double exponential_decay_term = std::exp(-beta_decay_lambda * frac);
    return beta_floor + (beta_init - beta_floor) * exponential_decay_term;
}

inline double peek_z_score(double x) {
    // if x is NaN or infinite, return 0.0
    if (std::isnan(x) || std::isinf(x)) {
        // print an error message
        std::cerr << "Error: NaN or infinite value encountered in peek_z_score." << std::endl;
        // print x
        std::cerr << "Value: " << x << std::endl;
        exit(1);
    }
    if (n < 2) {
        return 0.0;
    }
    double sigma = std::sqrt(m2); // Note: we use m2 directly for EMA variance
    constexpr double eps = 1e-8;
    return (x - mu) / (sigma + eps);
}

inline void update_stats(double x) {
    ++n;
    if (n <= 1) {
        // Initialize mu and m2 with the first value
        mu = x;
        m2 = 0.0;
    } else {
        // Calculate the EMA of the mean (mu)
        double delta1 = x - mu;
        mu    += delta1 * EMA_ALPHA;

        // Calculate the EMA of the variance (m2)
        double delta2 = x - mu;
        m2    = (1 - EMA_ALPHA) * (m2 + delta1 * delta2);
    }
}

} // namespace stats

#endif // STATS_H