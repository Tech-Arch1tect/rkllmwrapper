#include <cstdint>
#include "rkllm_wrapper.h"
#include "rkllm.h"
#include <cstring>
#include <string>
#include <mutex>
#include <condition_variable>
#include <cstdio>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <cerrno>
#include <chrono>
#include <thread>
#include <iostream>

static LLMHandle llmHandle = nullptr;
static std::mutex mtx;
static std::condition_variable cv;
static bool generation_finished = false;

struct InferenceData {
    std::string output;
    std::string fifo_path;
    int fifo_fd;
};

void writeToPersistentFifo(int fd, const char* text) {
    if (fd < 0) return;
    std::string out(text);
    out.append("\n");
    ssize_t written = write(fd, out.c_str(), out.size());
    if (written != static_cast<ssize_t>(out.size())) {
        perror("Failed to write to persistent FIFO");
    }
}

void unifiedCallback(RKLLMResult* result, void* userdata, LLMCallState state) {
    InferenceData* data = static_cast<InferenceData*>(userdata);

    if (state == RKLLM_RUN_FINISH) {
        writeToPersistentFifo(data->fifo_fd, "[[EOS]]");
        {
            std::lock_guard<std::mutex> lock(mtx);
            generation_finished = true;
        }
        cv.notify_one();
    } else if (state == RKLLM_RUN_ERROR) {
        std::printf("\nLLM run error\n");
        {
            std::lock_guard<std::mutex> lock(mtx);
            generation_finished = true;
        }
        cv.notify_one();
    } else {
        {
            std::lock_guard<std::mutex> lock(mtx);
            data->output += result->text;
        }
        writeToPersistentFifo(data->fifo_fd, result->text);
    }
}

int rkllm_init_simple(const char* model_path, int max_new_tokens, int max_context_len) {
    RKLLMParam param = rkllm_createDefaultParam();
    param.model_path = model_path;
    param.is_async = false;
    param.max_new_tokens = max_new_tokens;
    param.max_context_len = max_context_len;

    int ret = rkllm_init(&llmHandle, &param, unifiedCallback);
    if (ret != 0) {
        std::printf("rkllm_init failed with error: %d\n", ret);
    }
    return ret;
}

int rkllm_run_ex(const void *input, int input_mode, char* output, int output_size, size_t token_count, const char* fifo_path) {
    if (!llmHandle) return -1;

    RKLLMInput llmInput;
    if (input_mode == RKLLM_INPUT_PROMPT) {
        const char* promptStr = reinterpret_cast<const char*>(input);
        llmInput.input_type = static_cast<RKLLMInputType>(RKLLM_INPUT_PROMPT);
        llmInput.prompt_input = promptStr;
    } else if (input_mode == RKLLM_INPUT_TOKEN) {
        const int32_t* tokens = reinterpret_cast<const int32_t*>(input);
        llmInput.input_type = static_cast<RKLLMInputType>(RKLLM_INPUT_TOKEN);
        RKLLMTokenInput tokenInput;
        tokenInput.input_ids = const_cast<int32_t*>(tokens);
        tokenInput.n_tokens = token_count;
        llmInput.token_input = tokenInput;
    } else {
        return -1;
    }

    InferenceData* data = new InferenceData();
    data->output = "";
    data->fifo_path = fifo_path ? fifo_path : "";
    data->fifo_fd = -1;

    if (fifo_path && std::strlen(fifo_path) > 0) {
        int persistent_fd = open(fifo_path, O_WRONLY);
        if (persistent_fd == -1) {
            perror("Failed to open FIFO persistently");
            delete data;
            return -1;
        }
        data->fifo_fd = persistent_fd;
    }

    {
        std::unique_lock<std::mutex> lock(mtx);
        generation_finished = false;
    }

    RKLLMInferParam inferParams;
    inferParams.mode = RKLLM_INFER_GENERATE;
    inferParams.lora_params = nullptr;
    inferParams.prompt_cache_params = nullptr;
    inferParams.keep_history = 0;

    int ret = ::rkllm_run(llmHandle, &llmInput, &inferParams, static_cast<void*>(data));
    if (ret != 0) {
        if (data->fifo_fd >= 0) close(data->fifo_fd);
        delete data;
        return ret;
    }

    {
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait(lock, [] { return generation_finished; });
    }

    if (data->output.size() >= static_cast<size_t>(output_size)) {
        if (data->fifo_fd >= 0) close(data->fifo_fd);
        delete data;
        return -2;
    }
    std::strcpy(output, data->output.c_str());

    if (data->fifo_fd >= 0) close(data->fifo_fd);
    delete data;
    return 0;
}

void rkllm_destroy_simple() {
    if (llmHandle) {
        rkllm_destroy(llmHandle);
        llmHandle = nullptr;
    }
}
