#include "server.h"
#include <cerrno>
#include <chrono>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <fcntl.h>
#include <signal.h>
#include <sys/wait.h>
#include <curl/curl.h>

LlamaServer::LlamaServer(const std::string& model_path, int port, int n_gpu_layers, int ctx_size)
    : model_path_(model_path), port_(port), n_gpu_layers_(n_gpu_layers), ctx_size_(ctx_size) {}

LlamaServer::~LlamaServer() {
    if (is_running()) stop();
}

void LlamaServer::start() {
    pid_ = fork();
    if (pid_ < 0)
        throw std::runtime_error(std::string("fork() failed: ") + strerror(errno));

    if (pid_ == 0) {
        // Redirect all server output to log file — keeps our UI clean
        int fd = open(SERVER_LOG, O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (fd >= 0) {
            dup2(fd, STDOUT_FILENO);
            dup2(fd, STDERR_FILENO);
            close(fd);
        }

        std::vector<std::string> str_args = {
            "llama-server",
            "-m", model_path_,
            "--host", "127.0.0.1",
            "--port", std::to_string(port_),
            "-ngl", std::to_string(n_gpu_layers_),
            "-c", std::to_string(ctx_size_),
        };

        std::vector<char*> argv;
        for (auto& s : str_args) argv.push_back(const_cast<char*>(s.c_str()));
        argv.push_back(nullptr);

        execvp(argv[0], argv.data());
        // execvp failed — write to stderr (which is now the log file)
        std::cerr << "execvp(\"llama-server\") failed: " << strerror(errno) << "\n";
        std::cerr << "Is llama-server on PATH? Run: nix develop\n";
        _exit(1);
    }
}

bool LlamaServer::is_running() const {
    if (pid_ <= 0) return false;
    int status;
    pid_t r = waitpid(pid_, &status, WNOHANG);
    if (r == 0)    return true;
    if (r == pid_) pid_ = -1;
    return false;
}

static bool check_health(int port) {
    CURL* curl = curl_easy_init();
    if (!curl) return false;

    std::string url = "http://127.0.0.1:" + std::to_string(port) + "/health";
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 3L);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION,
        +[](char*, size_t s, size_t n, void*) -> size_t { return s * n; });

    CURLcode res = curl_easy_perform(curl);
    long code = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &code);
    curl_easy_cleanup(curl);
    return res == CURLE_OK && code == 200;
}

bool LlamaServer::wait_for_ready(int timeout_s) {
    auto start = std::chrono::steady_clock::now();

    // Brief pause then check it didn't immediately crash
    std::this_thread::sleep_for(std::chrono::seconds(1));
    if (!is_running()) return false;

    while (true) {
        int elapsed = static_cast<int>(std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::steady_clock::now() - start).count());

        if (elapsed >= timeout_s) {
            std::cout << "\n";
            return false;
        }
        if (!is_running()) {
            std::cout << "\n";
            return false;
        }

        // Inline elapsed counter — \r rewrites the same line cleanly
        std::cout << "\r  Loading model... " << elapsed << "s"
                  << "   (log: " << SERVER_LOG << ")   " << std::flush;

        if (check_health(port_)) {
            std::cout << "\n";
            return true;
        }

        std::this_thread::sleep_for(std::chrono::seconds(2));
    }
}

void LlamaServer::stop() {
    if (pid_ <= 0) return;
    kill(pid_, SIGTERM);
    int status;
    waitpid(pid_, &status, 0);
    pid_ = -1;
}
