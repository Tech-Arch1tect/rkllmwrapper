#ifndef RKLLM_WRAPPER_H
#define RKLLM_WRAPPER_H

#if defined(__cplusplus)
  #include <cstddef>
  #include <cstdint>
#else
  #include <stddef.h>
  #include <stdint.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int32_t max_new_tokens;
    int32_t max_context_len;
    int32_t top_k;
    float   top_p;
    float   temperature;
    float   repeat_penalty;
    float   frequency_penalty;
    float   presence_penalty;
    int32_t mirostat;
    float   mirostat_tau;
    float   mirostat_eta;
    int32_t n_keep;
    int     skip_special_token;
    int     num_cpus;
} RkllmOptions;

int  rkllmwrapper_init       (const char* model_path, const RkllmOptions* opts);
int  rkllm_run_ex            (const void* input, int input_mode, char* output, int output_size, size_t token_count, const char* fifo_path);
int  rkllmwrapper_is_running ();
int  rkllmwrapper_abort      ();
void rkllm_destroy_simple    ();

#ifdef __cplusplus
}
#endif

#endif