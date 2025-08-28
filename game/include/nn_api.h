#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

uint32_t parse_nn_params();

uint32_t init_nn(uint32_t input_dim, uint32_t output_dim, uint32_t hidden_dim, 
    uint32_t num_m_layers, uint32_t batch_size, uint32_t nn_type);

void predict_nn(uint32_t id, uint32_t nn_type, double* input_data, double* output_data, uint32_t batch_size);

void train_nn(uint32_t id, uint32_t nn_type, double* input_data, double* target_data, uint32_t batch_size);

void update_target_nn(uint32_t online_nn_id, uint32_t target_nn_id);

bool save_nn_model(uint32_t id, uint32_t nn_type, const char* dirname);

uint32_t load_nn_model(const char* dirname, uint32_t nn_type);

uint32_t randomize_weights(uint32_t id, uint32_t nn_type);

void reset_episode(uint32_t id, uint32_t nn_type);

#ifdef __cplusplus
}                           // End extern "C" block
#endif