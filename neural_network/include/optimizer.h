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

#endif