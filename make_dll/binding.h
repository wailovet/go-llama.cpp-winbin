#if defined(WIN32) || defined(_WIN32) || defined(_WIN32_) || defined(WIN64) || defined(_WIN64) || defined(_WIN64_)
#define DECEXT _declspec(dllexport)
#else
#define DECEXT
#endif

#include <stdbool.h>

#ifdef __cplusplus
extern "C"
{
#endif
    DECEXT void *load_model(const char *fname, int n_ctx, int n_parts, int n_seed, bool memory_f16, bool mlock, bool embedding);

    DECEXT void *llama_allocate_params(const char *prompt, int seed, int threads, int tokens,
                                       int top_k, float top_p, float temp, float repeat_penalty, int repeat_last_n, bool ignore_eos, bool memory_f16);

    DECEXT void llama_free_params(void *params_ptr);

    DECEXT void llama_free_model(void *state);
    
    DECEXT void llama_embedding(void *state_ptr, const char *text, float *embedding,int n_threads);
 
    DECEXT int llama_predict(void *params_ptr, void *state_pr);
 

    DECEXT int llama_get_embedding_size(void *state_ptr);
#ifdef __cplusplus
}
#endif