#ifndef IO_H
#define IO_H
#include <armadillo>
#include <layer_dense.h>

struct NNInfo_metadata {
    uint32_t input_dim;
    uint32_t output_dim;
    uint32_t hidden_dim;
    uint32_t num_m_layers;
    uint32_t batch_size;
    uint32_t nn_type;
};

bool write_model(const std::string& dirname,
    const std::vector<LayerDense>& layers, uint32_t input_dim, uint32_t output_dim, uint32_t hidden_dim, 
    uint32_t num_m_layers, uint32_t batch_size, uint32_t nn_type);

bool read_model(const std::string& dirname,
    std::vector<LayerDense>& layers, NNInfo_metadata& nn_info);

#endif