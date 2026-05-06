#include "config.h"
#include <stdexcept>
#include <yaml-cpp/yaml.h>

AppConfig load_config(const std::string& config_path) {
    YAML::Node root;
    try {
        root = YAML::LoadFile(config_path);
    } catch (const YAML::Exception& e) {
        throw std::runtime_error("Failed to load config: " + std::string(e.what()));
    }

    AppConfig cfg;

    auto defaults = root["defaults"]["backends"]["llama"];
    cfg.llama_version = defaults["version"].as<std::string>("b8971");
    cfg.n_gpu_layers  = defaults["args"]["n_gpu_layers"].as<int>(999);

    auto load_preset = [&](const std::string& yaml_key, const std::string& name) {
        if (!root[yaml_key]) return;
        auto n = root[yaml_key];
        BenchmarkPreset p;
        p.name        = name;
        p.prompt_set  = n["prompt_set"].as<std::string>(name);
        p.prompt      = n["prompt"].as<std::string>();
        p.max_tokens  = n["max_tokens"].as<int>(100);
        p.temperature = n["temperature"].as<float>(0.7f);
        p.iterations  = n["iterations"].as<int>(3);
        cfg.benchmarks[name] = p;
    };

    load_preset("quick_benchmark",       "quick");
    load_preset("standard_benchmark",    "standard");
    load_preset("performance_benchmark", "performance");

    if (root["benchmark_questions"] && root["benchmark_questions"]["hard_questions"]) {
        for (auto q : root["benchmark_questions"]["hard_questions"]) {
            HardQuestion hq;
            hq.prompt      = q["prompt"].as<std::string>();
            hq.max_tokens  = q["max_tokens"].as<int>(300);
            if (q["description"]) hq.description = q["description"].as<std::string>();
            cfg.hard_questions.push_back(hq);
        }
    }

    return cfg;
}
