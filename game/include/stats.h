#ifndef STATS_H
#define STATS_H

#include <cstddef>  // for std::size_t
#include <cmath>    // for std::sqrt

namespace stats {

inline size_t n    = 0;
inline double mu   = 0.0;
inline double m2   = 0.0;


inline constexpr double beta_init        = 1.0;
inline constexpr std::size_t beta_decay_steps = 100000;

inline constexpr double beta_floor = 0.01;

inline double current_beta() {
  double frac = std::min(1.0, double(n) / beta_decay_steps);
  return beta_floor + (beta_init - beta_floor) * (1.0 - frac);
}

inline double peek_z_score(double x) {
    if (n < 2) {
        return 0.0;
    }
    double sigma = std::sqrt(m2 / (n - 1));
    constexpr double eps = 1e-8;
    return (x - mu) / (sigma + eps);
}



inline void update_stats(double x) {
    ++n;
    double delta1 = x - mu;
    mu    += delta1 / n;
    double delta2 = x - mu;
    m2    += delta1 * delta2;
}



} // namespace stats


#endif // STATS_H