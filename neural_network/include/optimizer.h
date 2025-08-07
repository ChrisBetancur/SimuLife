#include <armadillo>
#include <layer_dense.h>

#ifndef OPTIMIZER_H
#define OPTIMIZER_H


class Optimizer_SGD {
    public:
        double m_learning_rate;
        double m_decay;
        double m_step;

        double m_momentum;

        Optimizer_SGD(double learning_rate, double decay = 0.0, double step = 0.0, double momentum = 0.0);

        void pre_update_params();

        void update(LayerDense& layer);

        void post_update_params();
};

class Optimizer_Adam {
public:
    double m_learning_rate;
    double m_decay;
    double m_step;

    double m_beta1;
    double m_beta2;
    double m_eps;

    Optimizer_Adam(double learning_rate = 0.001,
                   double beta1 = 0.9,
                   double beta2 = 0.999,
                   double eps = 1e-8,
                   double decay = 0.0,
                   double step = 0.0);

    void pre_update_params();
    void update(LayerDense &layer);
    void post_update_params();
};

#endif