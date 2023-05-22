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
using namespace std;

const LPCSTR llama_pipe_name = "\\\\.\\pipe\\llama_pipe";

void dumpParams(gpt_params *params_p)
{
    std::cout << "prompt: " << params_p->prompt << std::endl;
    std::cout << "n_batch: " << params_p->n_batch << std::endl;
    std::cout << "n_ctx: " << params_p->n_ctx << std::endl;
    std::cout << "temp: " << params_p->temp << std::endl;
}

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

    // dumpParams(params_p);

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
            // - take half of the last (n_ctx - n_keep) tokens and recompute the logits in batches
            if (n_past + (int)embd.size() > n_ctx)
            {
                const int n_left = n_past - params.n_keep;

                n_past = (std::max)(1, params.n_keep);

                // insert n_left/2 tokens at the start of embd from last_n_tokens
                embd.insert(embd.begin(), last_n_tokens.begin() + n_ctx - n_left / 2 - embd.size(), last_n_tokens.end() - embd.size());
            }

            // evaluate tokens in batches
            // embd is typically prepared beforehand to fit within a batch, but not always
            for (int i = 0; i < (int)embd.size(); i += params.n_batch)
            {
                int n_eval = (int)embd.size() - i;
                if (n_eval > params.n_batch)
                {
                    n_eval = params.n_batch;
                }
                if (llama_eval(ctx, &embd[i], n_eval, n_past, params.n_threads))
                {
                    fprintf(stderr, "%s : failed to eval\n", __func__);
                    CloseHandle(hPipe);
                    return 1;
                }
                n_past += n_eval;
            }
        }

        embd.clear();

        if ((int)embd_inp.size() <= n_consumed)
        {
            // out of user input, sample next token
            float temp = params.temp;
            const int32_t top_k = params.top_k <= 0 ? llama_n_vocab(ctx) : params.top_k;
            const float top_p = params.top_p;
            const float tfs_z = params.tfs_z;
            const float typical_p = params.typical_p;
            const int32_t repeat_last_n = params.repeat_last_n < 0 ? n_ctx : params.repeat_last_n;
            const float repeat_penalty = params.repeat_penalty;
            const float alpha_presence = params.presence_penalty;
            const float alpha_frequency = params.frequency_penalty;
            const int mirostat = params.mirostat;
            const float mirostat_tau = params.mirostat_tau;
            const float mirostat_eta = params.mirostat_eta;
            const bool penalize_nl = params.penalize_nl;

            llama_token id = 0;

            {
                auto logits = llama_get_logits(ctx);
                auto n_vocab = llama_n_vocab(ctx);

                // Apply params.logit_bias map
                for (auto it = params.logit_bias.begin(); it != params.logit_bias.end(); it++)
                {
                    logits[it->first] += it->second;
                }
                std::vector<llama_token_data> candidates;
                candidates.reserve(n_vocab);
                for (llama_token token_id = 0; token_id < n_vocab; token_id++)
                {
                    candidates.emplace_back(llama_token_data{token_id, logits[token_id], 0.0f});
                }
                llama_token_data_array candidates_p = {candidates.data(), candidates.size(), false};

                // Apply penalties
                float nl_logit = logits[llama_token_nl()];

                auto last_n_repeat = (std::min)((std::min)((int)last_n_tokens.size(), repeat_last_n), n_ctx);

                llama_sample_repetition_penalty(ctx, &candidates_p,
                                                last_n_tokens.data() + last_n_tokens.size() - last_n_repeat,
                                                last_n_repeat, repeat_penalty);
                llama_sample_frequency_and_presence_penalties(ctx, &candidates_p,
                                                              last_n_tokens.data() + last_n_tokens.size() - last_n_repeat,
                                                              last_n_repeat, alpha_frequency, alpha_presence);
                if (!penalize_nl)
                {
                    logits[llama_token_nl()] = nl_logit;
                }

                if (temp <= 0)
                {
                    temp = 0.8;
                }

                {
                    if (mirostat == 1)
                    {
                        static float mirostat_mu = 2.0f * mirostat_tau;
                        const int mirostat_m = 100;
                        llama_sample_temperature(ctx, &candidates_p, temp);
                        id = llama_sample_token_mirostat(ctx, &candidates_p, mirostat_tau, mirostat_eta, mirostat_m, &mirostat_mu);
                    }
                    else if (mirostat == 2)
                    {
                        static float mirostat_mu = 2.0f * mirostat_tau;
                        llama_sample_temperature(ctx, &candidates_p, temp);
                        id = llama_sample_token_mirostat_v2(ctx, &candidates_p, mirostat_tau, mirostat_eta, &mirostat_mu);
                    }
                    else
                    {
                        // Temperature sampling
                        llama_sample_top_k(ctx, &candidates_p, top_k, 1);
                        llama_sample_tail_free(ctx, &candidates_p, tfs_z, 1);
                        llama_sample_typical(ctx, &candidates_p, typical_p, 1);
                        llama_sample_top_p(ctx, &candidates_p, top_p, 1);
                        llama_sample_temperature(ctx, &candidates_p, temp);
                        id = llama_sample_token(ctx, &candidates_p);
                    }
                }

                last_n_tokens.erase(last_n_tokens.begin());
                last_n_tokens.push_back(id);
            }

            // add it to the context
            embd.push_back(id);

            // decrement remaining sampling budget
            --n_remain;

            llama_pipe_res += llama_token_to_str(ctx, id);
            int embd_size = embd.size();
            auto data = Json::object{
                {"content", llama_pipe_res},
                {"tokens_size", embd_size},
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
        }
        else
        {
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
        if (!embd.empty() && embd.back() == llama_token_eos())
        {
            break;
        }
    }

    // printf("CloseHandle\n");
    CloseHandle(hPipe);
    // printf("CloseHandle ok\n");
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

void *llama_allocate_params(const char *prompt, int seed, int threads, int tokens, int top_k,
                            float top_p, float temp, float repeat_penalty, int repeat_last_n, bool ignore_eos, bool memory_f16, int batch)
{
    // std::cout << "llama_allocate_params:"
    //           << "[seed]:" << seed << "  ";
    // std::cout << "[threads]:" << threads << "  ";
    // std::cout << "[tokens]:" << tokens << "  ";
    // std::cout << "[top_k]:" << top_k << "  ";
    // std::cout << "[top_p]:" << top_p << "  ";
    // std::cout << "[temp]:" << temp << "  ";
    // std::cout << "[repeat_penalty]:" << repeat_penalty << "  ";
    // std::cout << "[repeat_last_n]:" << repeat_last_n << "  ";
    // std::cout << "[ignore_eos]:" << ignore_eos << "  ";
    // std::cout << "[memory_f16]:" << memory_f16 << "  ";
    // std::cout << "[batch]:" << batch << "  ";

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
    params->n_batch = batch;
    params->penalize_nl = false;

    if (ignore_eos)
    {
        params->logit_bias[llama_token_eos()] = -INFINITY;
    }

    return params;
}

void *load_model(const char *fname, int n_ctx, int n_parts, int n_seed, bool memory_f16, bool mlock, int n_gpu_layers)
{
    llama_init_backend();

    // load the model
    auto lparams = llama_context_default_params();

    lparams.n_ctx = n_ctx;
    // lparams.n_parts = n_parts;
    lparams.seed = n_seed;
    // lparams.f16_kv = memory_f16;
    lparams.f16_kv = true;
    lparams.use_mlock = mlock;
    // lparams.embedding = embedding;

    lparams.n_gpu_layers = n_gpu_layers;

    return llama_init_from_file(fname, lparams);
}
