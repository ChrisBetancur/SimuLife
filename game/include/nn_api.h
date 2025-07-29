#ifdef __cplusplus          // If compiling as C++...
extern "C" {                // ...disable C++ name mangling
#endif

#include <stdint.h>

uint32_t init_nn(uint32_t input_dim, uint32_t output_dim, uint32_t hidden_dim, 
    uint32_t num_m_layers, uint32_t batch_size, uint32_t nn_type);

double* predict_nn(uint32_t id, uint32_t nn_type, double* input_data);

void train_nn(uint32_t id, uint32_t nn_type, double* target_data);

void update_target_nn(uint32_t online_nn_id, uint32_t target_nn_id);

bool save_nn_model(uint32_t id, uint32_t nn_type, const char* dirname);

uint32_t load_nn_model(const char* dirname, uint32_t nn_type);

uint32_t randomize_weights(uint32_t id, uint32_t nn_type);

#ifdef __cplusplus
}                           // End extern "C" block
#endif