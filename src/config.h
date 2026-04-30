#pragma once
#include <map>
#include <string>
#include <vector>

struct BenchmarkPreset {
    std::string name;
    std::string prompt_set;
    std::string prompt;
    int max_tokens = 100;
    float temperature = 0.7f;
    int iterations = 3;
};

struct HardQuestion {
    std::string prompt;
    int max_tokens = 300;
    std::string description;
};

struct AppConfig {
    std::string llama_version;
    int n_gpu_layers = 999;
    std::map<std::string, BenchmarkPreset> benchmarks;
    std::vector<HardQuestion> hard_questions;
};

AppConfig load_config(const std::string& config_path);
