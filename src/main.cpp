#include <CLI/CLI.hpp>
#include "benchmark.h"
#include "config.h"
#include "models.h"
#include "server.h"
#include <filesystem>
#include <iostream>
#include <memory>

// Walk up from the binary's directory until config/models.yaml is found.
static std::filesystem::path find_project_root(const char* argv0) {
    std::error_code ec;
    auto root = std::filesystem::canonical(argv0, ec);
    if (ec) root = std::filesystem::current_path();

    root = root.parent_path();
    for (int depth = 0; depth < 4; depth++) {
        if (std::filesystem::exists(root / "config" / "models.yaml")) return root;
        auto parent = root.parent_path();
        if (parent == root) break;
        root = parent;
    }
    return std::filesystem::current_path();
}

int main(int argc, char* argv[]) {
    CLI::App app{"Kepler — LLM benchmarking for macOS Metal"};
    app.footer("Benchmark types: quick | standard | performance | hard");

    std::string model_path;
    std::string benchmark_type = "standard";
    int         port           = 8080;
    int         ctx_size       = 4096;
    bool        list_models    = false;
    bool        no_serve       = false;

    app.add_option("-m,--model",      model_path,      "Path to .gguf model file");
    app.add_option("-b,--benchmark",  benchmark_type,  "Benchmark preset")->default_val("standard");
    app.add_option("-p,--port",       port,            "llama-server port")->default_val(8080);
    app.add_option("-c,--ctx-size",   ctx_size,        "Context size")->default_val(4096);
    app.add_flag("--list-models",     list_models,     "List available models and exit");
    app.add_flag("--no-serve",        no_serve,        "Skip server startup (testing only)");

    CLI11_PARSE(app, argc, argv);

    auto root       = find_project_root(argv[0]);
    auto models_dir = root / "models";
    auto config_path = root / "config" / "models.yaml";
    auto perf_dir   = root / "perf";

    // ── load config ──────────────────────────────────────────────────────
    AppConfig cfg;
    try {
        cfg = load_config(config_path.string());
    } catch (const std::exception& e) {
        std::cerr << "[kepler] " << e.what() << "\n";
        return 1;
    }

    // ── list models ──────────────────────────────────────────────────────
    if (list_models) {
        auto models = find_gguf_models(models_dir);
        if (models.empty()) {
            std::cout << "No .gguf models found in " << models_dir.string() << "\n";
        } else {
            std::cout << "Available models in " << models_dir.string() << ":\n";
            for (auto& m : models) {
                auto bytes = std::filesystem::file_size(m);
                std::cout << "  " << m.filename().string()
                          << "  (" << (bytes / (1024 * 1024)) << " MB)\n";
            }
        }
        return 0;
    }

    // ── select model ─────────────────────────────────────────────────────
    std::filesystem::path selected;
    if (!model_path.empty()) {
        selected = model_path;
        if (!std::filesystem::exists(selected)) {
            std::cerr << "[kepler] Model not found: " << model_path << "\n";
            return 1;
        }
    } else {
        try {
            auto models = find_gguf_models(models_dir);
            selected    = select_model_interactive(models);
        } catch (const std::exception& e) {
            std::cerr << "[kepler] " << e.what() << "\n";
            return 1;
        }
    }

    std::cout << "\n";
    std::cout << "[kepler] model:     " << selected.filename().string() << "\n";
    std::cout << "[kepler] benchmark: " << benchmark_type << "\n";
    std::cout << "[kepler] port:      " << port << "\n\n";

    // ── start server ─────────────────────────────────────────────────────
    std::unique_ptr<LlamaServer> server;
    if (!no_serve) {
        std::error_code ec;
        auto abs_model = std::filesystem::canonical(selected, ec);
        if (ec) abs_model = selected;

        server = std::make_unique<LlamaServer>(abs_model.string(), port, cfg.n_gpu_layers, ctx_size);

        std::cout << "[1/4] Starting llama-server...\n";
        try {
            server->start();
        } catch (const std::exception& e) {
            std::cerr << "[kepler] " << e.what() << "\n";
            return 1;
        }

        std::cout << "[2/4] Waiting for server to be ready (model loading into Metal)...\n";
        if (!server->wait_for_ready(120)) {
            std::cerr << "[kepler] Server did not become ready within 120s\n";
            return 1;
        }
        std::cout << "[2/4] Server ready.\n\n";
    }

    // ── run benchmarks ───────────────────────────────────────────────────
    std::cout << "[3/4] Running benchmarks...\n";
    std::vector<BenchmarkResult> results;

    try {
        if (benchmark_type == "hard") {
            if (cfg.hard_questions.empty()) {
                std::cerr << "[kepler] No hard questions found in config\n";
                return 1;
            }
            for (size_t i = 0; i < cfg.hard_questions.size(); i++) {
                auto& q = cfg.hard_questions[i];
                std::cout << "\n  Hard question " << (i + 1) << "/" << cfg.hard_questions.size();
                if (!q.description.empty()) std::cout << ": " << q.description;
                std::cout << "\n";

                BenchmarkConfig bcfg;
                bcfg.prompt_set  = "hard_question_" + std::to_string(i + 1);
                bcfg.prompt      = q.prompt;
                bcfg.max_tokens  = q.max_tokens;
                bcfg.temperature = 0.7f;
                bcfg.iterations  = 3;
                bcfg.host        = "localhost";
                bcfg.port        = port;

                results.push_back(run_benchmark(selected.string(), bcfg));
            }
        } else {
            auto it = cfg.benchmarks.find(benchmark_type);
            if (it == cfg.benchmarks.end()) {
                std::cerr << "[kepler] Unknown benchmark type: " << benchmark_type << "\n";
                std::cerr << "[kepler] Available: quick, standard, performance, hard\n";
                return 1;
            }

            auto& preset = it->second;
            BenchmarkConfig bcfg;
            bcfg.prompt_set  = preset.prompt_set;
            bcfg.prompt      = preset.prompt;
            bcfg.max_tokens  = preset.max_tokens;
            bcfg.temperature = preset.temperature;
            bcfg.iterations  = preset.iterations;
            bcfg.host        = "localhost";
            bcfg.port        = port;

            results.push_back(run_benchmark(selected.string(), bcfg));
        }
    } catch (const std::exception& e) {
        std::cerr << "[kepler] Benchmark failed: " << e.what() << "\n";
        return 1;
    }

    // ── save and print ───────────────────────────────────────────────────
    std::cout << "\n[4/4] Saving results...\n";
    for (auto& r : results) {
        print_summary(r);
        save_result(r, perf_dir);
    }

    return 0;
}
