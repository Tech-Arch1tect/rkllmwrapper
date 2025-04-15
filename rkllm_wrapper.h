#ifndef RKLLM_WRAPPER_H
#define RKLLM_WRAPPER_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h> 

int rkllm_init_simple(const char* model_path, int max_new_tokens, int max_context_len);

int rkllm_run_simple(const char* prompt, int input_mode, char* output, int output_size);

int rkllm_run_simple_with_fifo(void* input, int input_mode, const char* fifo, char* output, int output_size, size_t token_count);

void rkllm_destroy_simple();

#ifdef __cplusplus
}
#endif

#endif
