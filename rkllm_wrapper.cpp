#include "rkllm_wrapper.h"
#include "rkllm.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <mutex>
#include <string>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>

static LLMHandle  llmHandle = nullptr;
static std::mutex mtx;
static std::condition_variable cv;
static bool       generation_finished = false;

struct InferenceData {
    std::string output;
    std::string fifo_path;
    int fifo_fd = -1;
};

static void writeToPersistentFifo(int fd, const char* text) {
    if (fd < 0) return;
    std::string out(text);
    out.push_back('\n');
    ssize_t written = write(fd, out.c_str(), out.size());
    if (written != static_cast<ssize_t>(out.size()))
        perror("rkllm_wrapper: write to FIFO failed");
}

static void unifiedCallback(RKLLMResult* result, void* userdata, LLMCallState state) {
    auto* data = static_cast<InferenceData*>(userdata);

    switch (state) {
        case RKLLM_RUN_FINISH:
            writeToPersistentFifo(data->fifo_fd, "[[EOS]]");
            {
                std::lock_guard<std::mutex> lk(mtx);
                generation_finished = true;
            }
            cv.notify_one();
            break;
        case RKLLM_RUN_ERROR:
            std::fprintf(stderr, "rkllm_wrapper: LLM run error\n");
            {
                std::lock_guard<std::mutex> lk(mtx);
                generation_finished = true;
            }
            cv.notify_one();
            break;
        default:
            {
                std::lock_guard<std::mutex> lk(mtx);
                data->output += result->text;
            }
            writeToPersistentFifo(data->fifo_fd, result->text);
            break;
    }
}

int rkllmwrapper_init(const char* model_path, const RkllmOptions* opts) {
    if (!opts) return -1;

    RKLLMParam param = rkllm_createDefaultParam();
    param.model_path      = model_path;
    param.is_async        = false;
    param.max_new_tokens  = opts->max_new_tokens;
    param.max_context_len = opts->max_context_len;

    if (opts->top_k       > 0)   param.top_k       = opts->top_k;
    if (opts->top_p       > 0.f) param.top_p       = opts->top_p;
    if (opts->temperature > 0.f) param.temperature = opts->temperature;
    if (opts->repeat_penalty     > 0.f) param.repeat_penalty   = opts->repeat_penalty;
    if (opts->frequency_penalty  > 0.f) param.frequency_penalty = opts->frequency_penalty;
    if (opts->presence_penalty   > 0.f) param.presence_penalty  = opts->presence_penalty;
    if (opts->mirostat          >= 0)   param.mirostat         = opts->mirostat;
    if (opts->mirostat_tau       > 0.f) param.mirostat_tau      = opts->mirostat_tau;
    if (opts->mirostat_eta       > 0.f) param.mirostat_eta      = opts->mirostat_eta;
    if (opts->n_keep             > 0)   param.n_keep           = opts->n_keep;
    param.skip_special_token = opts->skip_special_token ? true : false;

    if (opts->num_cpus > 0) {
        int N = opts->num_cpus;
        uint32_t mask = (N >= 32 ? 0xFFFFFFFFu : ((1u << N) - 1));
        param.extend_param.enabled_cpus_mask = mask;
        param.extend_param.enabled_cpus_num  = N;
    }

    int ret = rkllm_init(&llmHandle, &param, unifiedCallback);
    if (ret)
        std::fprintf(stderr, "rkllmwrapper_init: rkllm_init failed (%d)\n", ret);
    return ret;
}

int rkllm_run_ex(const void* input, int input_mode, char* output, int output_size, size_t token_count, const char* fifo_path) {
    if (!llmHandle) return -1;

    RKLLMInput llmInput{};
    int32_t*   cTokens = nullptr;

    if (input_mode == RKLLM_INPUT_PROMPT) {
        llmInput.input_type   = RKLLM_INPUT_PROMPT;
        llmInput.prompt_input = static_cast<const char*>(input);
    } else if (input_mode == RKLLM_INPUT_TOKEN) {
        cTokens = static_cast<int32_t*>(std::malloc(token_count * sizeof(int32_t)));
        if (!cTokens) {
            std::fprintf(stderr, "rkllm_wrapper: token malloc failed\n");
            return -1;
        }
        std::memcpy(cTokens, input, token_count * sizeof(int32_t));
        llmInput.input_type = RKLLM_INPUT_TOKEN;
        RKLLMTokenInput tokenInput{cTokens, token_count};
        llmInput.token_input = tokenInput;
    } else {
        return -1;
    }

    auto* data = new InferenceData{};

    if (fifo_path && *fifo_path) {
        int fd = open(fifo_path, O_WRONLY);
        if (fd == -1) {
            perror("rkllm_wrapper: open FIFO failed");
            delete data;
            if (cTokens) std::free(cTokens);
            return -1;
        }
        data->fifo_path = fifo_path;
        data->fifo_fd   = fd;
    }

    {
        std::lock_guard<std::mutex> lk(mtx);
        generation_finished = false;
    }

    RKLLMInferParam inferParams{};
    inferParams.mode = RKLLM_INFER_GENERATE;

    int ret = rkllm_run(llmHandle, &llmInput, &inferParams, data);
    if (ret) {
        if (data->fifo_fd >= 0) close(data->fifo_fd);
        delete data;
        if (cTokens) std::free(cTokens);
        return ret;
    }

    {
        std::unique_lock<std::mutex> ulk(mtx);
        cv.wait(ulk, [] { return generation_finished; });
    }

    if (data->output.size() >= static_cast<size_t>(output_size)) {
        if (data->fifo_fd >= 0) close(data->fifo_fd);
        delete data;
        if (cTokens) std::free(cTokens);
        return -2;
    }

    std::strcpy(output, data->output.c_str());

    if (data->fifo_fd >= 0) close(data->fifo_fd);
    delete data;
    if (cTokens) std::free(cTokens);

    return 0;
}

int rkllmwrapper_is_running() {
    if (!llmHandle) return -1;
    return rkllm_is_running(llmHandle);
}

int rkllmwrapper_abort() {
    if (!llmHandle) return -1;

    int ret = rkllm_abort(llmHandle);

    {
        std::lock_guard<std::mutex> lk(mtx);
        generation_finished = true;
    }
    cv.notify_one();

    return ret;
}

void rkllm_destroy_simple() {
    if (llmHandle) {
        rkllm_destroy(llmHandle);
        llmHandle = nullptr;
    }
}
