#include "common.h"
#include "llama.h"
#include "binding.h"

#include <cassert>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <Windows.h>
#include "json11.hpp"

#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__))
#include <signal.h>
#include <unistd.h>
#elif defined(_WIN32)
#include <signal.h>
#endif

#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__)) || defined(_WIN32)
void sigint_handler(int signo)
{
    if (signo == SIGINT)
    {
        _exit(130);
    }
}
#endif

using namespace json11;
 
const LPCSTR llama_pipe_name = "\\\\.\\pipe\\llama_pipe";

int llama_predict(void *params_ptr, void *state_pr)
{

    if (!WaitNamedPipeA(llama_pipe_name, 1000))
    {
        std::cout << "等待管道超时" << std::endl;
        return 0;
    }
    HANDLE hPipe; // 管道句柄
    // step2:连接管道
    hPipe = CreateFileA(llama_pipe_name,
                        GENERIC_READ | GENERIC_WRITE,
                        0,
                        NULL,
                        OPEN_EXISTING,
                        FILE_ATTRIBUTE_NORMAL,
                        NULL);

    if (INVALID_HANDLE_VALUE == hPipe)
    {
        // 成功
        std::cout << "连接管道失败" << GetLastError() << std::endl;
        return 0;
    }

    std::string llama_pipe_res = "";

    gpt_params *params_p = (gpt_params *)params_ptr;
    llama_context *ctx = (llama_context *)state_pr;

    gpt_params params = *params_p;

    if (params.seed <= 0)
    {
        params.seed = time(NULL);
    }

    std::mt19937 rng(params.seed);

    // Add a space in front of the first character to match OG llama tokenizer behavior
    params.prompt.insert(0, 1, ' ');

    // tokenize the prompt
    auto embd_inp = ::llama_tokenize(ctx, params.prompt, true);

    const int n_ctx = llama_n_ctx(ctx);

    // number of tokens to keep when resetting context
    if (params.n_keep < 0 || params.n_keep > (int)embd_inp.size() || params.instruct)
    {
        params.n_keep = (int)embd_inp.size();
    }

    // determine newline token
    auto llama_token_newline = ::llama_tokenize(ctx, "\n", false);

    // TODO: replace with ring-buffer
    std::vector<llama_token> last_n_tokens(n_ctx);
    std::fill(last_n_tokens.begin(), last_n_tokens.end(), 0);

    int n_past = 0;
    int n_remain = params.n_predict;
    int n_consumed = 0;

    std::vector<llama_token> embd;
    std::string res = "";

    while (n_remain != 0)
    {
        // predict
        if (embd.size() > 0)
        {
            // infinite text generation via context swapping
            // if we run out of context:
            // - take the n_keep first tokens from the original prompt (via n_past)
            // - take half of the last (n_ctx - n_keep) tokens and recompute the logits in a batch
            if (n_past + (int)embd.size() > n_ctx)
            {
                const int n_left = n_past - params.n_keep;

                n_past = params.n_keep;

                // insert n_left/2 tokens at the start of embd from last_n_tokens
                embd.insert(embd.begin(), last_n_tokens.begin() + n_ctx - n_left / 2 - embd.size(), last_n_tokens.end() - embd.size());
            }

            if (llama_eval(ctx, embd.data(), embd.size(), n_past, params.n_threads))
            {
                fprintf(stderr, "%s : failed to eval\n", __func__);
                CloseHandle(hPipe);
                return 1;
            }
        }

        n_past += embd.size();
        embd.clear();

        if ((int)embd_inp.size() <= n_consumed)
        {
            // out of user input, sample next token
            const int32_t top_k = params.top_k;
            const float top_p = params.top_p;
            const float temp = params.temp;
            const float repeat_penalty = params.repeat_penalty;

            llama_token id = 0;

            {
                auto logits = llama_get_logits(ctx);

                if (params.ignore_eos)
                {
                    logits[llama_token_eos()] = 0;
                }

                id = llama_sample_top_p_top_k(ctx,
                                              last_n_tokens.data() + n_ctx - params.repeat_last_n,
                                              params.repeat_last_n, top_k, top_p, temp, repeat_penalty);

                last_n_tokens.erase(last_n_tokens.begin());
                last_n_tokens.push_back(id);
            }

            // add it to the context
            embd.push_back(id);

            llama_pipe_res += llama_token_to_str(ctx, id);
            auto data = Json::object{
                {"content", llama_pipe_res},
            };
            std::string pre_send_data = Json(data).dump() + "\n";
            // std::cout << "llama_pipe_res:" << pre_send_data << std::endl;
            if (!WriteFile(hPipe,                 // 管道句柄
                           pre_send_data.c_str(), // 要写入的数据
                           pre_send_data.size(),  // 要写入的数据的长度
                           NULL,                  // 实际写入的数据的长度
                           NULL))
            {
                std::cout << "写入管道失败" << GetLastError() << std::endl;
                CloseHandle(hPipe);
                return 0;
            } 

            // decrement remaining sampling budget
            --n_remain;
        }
        else
        {
            // some user input remains from prompt or interaction, forward it to processing
            while ((int)embd_inp.size() > n_consumed)
            {
                embd.push_back(embd_inp[n_consumed]);

                last_n_tokens.erase(last_n_tokens.begin());
                last_n_tokens.push_back(embd_inp[n_consumed]);
                ++n_consumed;
                if ((int)embd.size() >= params.n_batch)
                {
                    break;
                }
            }
        }

        for (auto id : embd)
        {
            res += llama_token_to_str(ctx, id);
        }

        // end of text token
        if (embd.back() == llama_token_eos())
        {
            break;
        }
    }
 

    CloseHandle(hPipe);
    return 0;
}

void llama_free_model(void *state_ptr)
{
    llama_context *ctx = (llama_context *)state_ptr;
    llama_free(ctx);
}

void llama_free_params(void *params_ptr)
{
    gpt_params *params = (gpt_params *)params_ptr;
    delete params;
}

char *new_chars(int size)
{
    char *chars = new char[size];
    return chars;
}

void *llama_allocate_params(const char *prompt, int seed, int threads, int tokens, int top_k,
                            float top_p, float temp, float repeat_penalty, int repeat_last_n, bool ignore_eos, bool memory_f16)
{
    gpt_params *params = new gpt_params;
    params->seed = seed;
    params->n_threads = threads;
    params->n_predict = tokens;
    params->repeat_last_n = repeat_last_n;

    params->top_k = top_k;
    params->top_p = top_p;
    params->memory_f16 = memory_f16;
    params->temp = temp;
    params->repeat_penalty = repeat_penalty;

    params->prompt = prompt;
    params->ignore_eos = ignore_eos;

    return params;
}

void *load_model(const char *fname, int n_ctx, int n_parts, int n_seed, bool memory_f16, bool mlock)
{
    // load the model
    auto lparams = llama_context_default_params();

    lparams.n_ctx = n_ctx;
    lparams.n_parts = n_parts;
    lparams.seed = n_seed;
    lparams.f16_kv = memory_f16;
    lparams.use_mlock = mlock;

    return llama_init_from_file(fname, lparams);
}
