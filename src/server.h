#pragma once
#include <string>
#include <unistd.h>

inline constexpr const char* SERVER_LOG = "/tmp/kepler-llama-server.log";

class LlamaServer {
public:
    LlamaServer(const std::string& model_path,
                int port,
                int n_gpu_layers = 999,
                int ctx_size     = 4096);
    ~LlamaServer();

    void start();
    bool wait_for_ready(int timeout_s = 120);
    void stop();
    bool is_running() const;
    int  port() const { return port_; }

private:
    std::string   model_path_;
    int           port_;
    int           n_gpu_layers_;
    int           ctx_size_;
    mutable pid_t pid_ = -1;
};
