#pragma once
#include <filesystem>
#include <string>
#include <vector>

struct BenchmarkConfig {
    std::string prompt_set;
    std::string prompt;
    int         max_tokens  = 100;
    float       temperature = 0.7f;
    int         iterations  = 3;
    std::string host        = "localhost";
    int         port        = 8080;
};

struct BenchmarkIteration {
    double      wall_s;
    std::string output_text;
    int         prompt_tokens;
    int         generated_tokens;
    double      tok_per_s;
    double      generation_tok_per_s;
    double      ttft_ms;
    double      prefill_ms;
    double      generation_ms;
};

struct SystemInfo {
    std::string platform;
    std::string architecture;
    std::string processor;
};

struct BenchmarkSummary {
    double median_wall_s;
    double median_tok_per_s;
    double median_ttft_ms;
    double median_generation_tok_per_s;
};

struct BenchmarkResult {
    std::string                      timestamp;
    std::string                      repo_id;
    std::string                      model_ref;
    std::string                      engine      = "llama-server";
    SystemInfo                       system_info;
    BenchmarkConfig                  config;
    std::vector<BenchmarkIteration>  iterations;
    BenchmarkSummary                 summary;
    std::string                      mode        = "text-only";
    std::string                      environment = "stable";
};

SystemInfo      get_system_info();
BenchmarkIteration run_once(const BenchmarkConfig& cfg);
BenchmarkResult    run_benchmark(const std::string& model_path, const BenchmarkConfig& cfg);
void               save_result(const BenchmarkResult& result, const std::filesystem::path& perf_dir);
void               print_summary(const BenchmarkResult& result);
