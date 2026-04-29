#include <CLI/CLI.hpp>
#include "benchmark.h"
#include "config.h"
#include "models.h"
#include "server.h"
#include "ui.h"
#include <filesystem>
#include <iostream>
#include <memory>

static std::filesystem::path find_project_root(const char* argv0) {
    std::error_code ec;
    auto root = std::filesystem::canonical(argv0, ec);
    if (ec) root = std::filesystem::current_path();
    root = root.parent_path();
    for (int i = 0; i < 4; i++) {
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

    std::string model_arg;
    std::string benchmark_arg;       // empty = not set, show interactive picker
    int         port       = 8080;
    int         ctx_size   = 4096;
    bool        list_models = false;
    bool        no_serve    = false;

    app.add_option("-m,--model",     model_arg,     "Path to .gguf model file (skips picker)");
    app.add_option("-b,--benchmark", benchmark_arg, "Benchmark preset (skips picker)");
    app.add_option("-p,--port",      port,          "llama-server port")->default_val(8080);
    app.add_option("-c,--ctx-size",  ctx_size,      "Context size")->default_val(4096);
    app.add_flag("--list-models",    list_models,   "List available models and exit");
    app.add_flag("--no-serve",       no_serve,      "Skip server startup (testing only)");

    CLI11_PARSE(app, argc, argv);

    auto root        = find_project_root(argv[0]);
    auto models_dir  = root / "models";
    auto config_path = root / "config" / "models.yaml";
    auto perf_dir    = root / "perf";

    // ── load config ──────────────────────────────────────────────────────
    AppConfig cfg;
    try {
        cfg = load_config(config_path.string());
    } catch (const std::exception& e) {
        ui::error(e.what());
        return 1;
    }

    // ── list models ──────────────────────────────────────────────────────
    if (list_models) {
        ui::print_banner();
        auto models = find_gguf_models(models_dir);
        if (models.empty()) {
            ui::warn("No .gguf models found in " + models_dir.string());
        } else {
            std::cout << "  Available models:\n\n";
            for (auto& m : models) {
                auto bytes = std::filesystem::file_size(m);
                std::cout << "  " << ui::CYAN << "▸" << ui::RESET << " "
                          << m.filename().string()
                          << ui::DIM << "  (" << (bytes / (1024 * 1024)) << " MB)"
                          << ui::RESET << "\n";
            }
            std::cout << "\n";
        }
        return 0;
    }

    // ── phase 1: interactive selections (before tracker clears screen) ───
    // All user prompts happen here so the tracker's clear+redraw never
    // wipes an in-progress interactive prompt.

    ui::print_banner();

    // Model selection
    std::filesystem::path selected;
    if (!model_arg.empty()) {
        selected = model_arg;
        if (!std::filesystem::exists(selected)) {
            ui::error("Model not found: " + model_arg);
            return 1;
        }
        ui::success("Model: " + selected.filename().string());
    } else {
        auto models = find_gguf_models(models_dir);
        try {
            selected = select_model_interactive(models);
            ui::success("Model: " + selected.filename().string());
        } catch (const std::exception& e) {
            ui::error(e.what());
            return 1;
        }
    }

    std::cout << "\n";

    // Benchmark selection
    std::string benchmark_type = benchmark_arg;
    if (benchmark_type.empty()) {
        benchmark_type = ui::pick_benchmark();
    }
    ui::success("Benchmark: " + benchmark_type);
    std::cout << "\n";

    // ── phase 2: automated pipeline with workflow tracker ────────────────
    ui::WorkflowTracker tracker;
    tracker.add("Server Start");
    tracker.add("Health Check");
    tracker.add("Benchmarking");
    tracker.add("Save Results");
    tracker.redraw();

    // ── server start ─────────────────────────────────────────────────────
    std::unique_ptr<LlamaServer> server;
    if (!no_serve) {
        tracker.start("Server Start");
        try {
            std::error_code ec;
            auto abs_model = std::filesystem::canonical(selected, ec);
            if (ec) abs_model = selected;

            server = std::make_unique<LlamaServer>(
                abs_model.string(), port, cfg.n_gpu_layers, ctx_size);
            server->start();
            tracker.done("Server Start");
        } catch (const std::exception& e) {
            tracker.fail("Server Start");
            ui::error(e.what());
            return 1;
        }

        // ── health check ─────────────────────────────────────────────────
        tracker.start("Health Check");
        if (!server->wait_for_ready(120)) {
            tracker.fail("Health Check");
            ui::error("Server did not become ready within 120s");
            ui::warn("Check logs: " + std::string(SERVER_LOG));
            return 1;
        }
        tracker.done("Health Check");
    } else {
        tracker.done("Server Start");
        tracker.done("Health Check");
    }

    // ── benchmarking ─────────────────────────────────────────────────────
    tracker.start("Benchmarking");
    std::vector<BenchmarkResult> results;
    try {
        if (benchmark_type == "hard") {
            if (cfg.hard_questions.empty()) {
                tracker.fail("Benchmarking");
                ui::error("No hard questions found in config");
                return 1;
            }
            for (size_t i = 0; i < cfg.hard_questions.size(); i++) {
                auto& q = cfg.hard_questions[i];
                ui::info("Hard question " + std::to_string(i + 1) + "/" +
                         std::to_string(cfg.hard_questions.size()) +
                         (q.description.empty() ? "" : ": " + q.description));

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
                tracker.fail("Benchmarking");
                ui::error("Unknown benchmark type: " + benchmark_type);
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
        tracker.done("Benchmarking");
    } catch (const std::exception& e) {
        tracker.fail("Benchmarking");
        ui::error(e.what());
        return 1;
    }

    // ── save results ─────────────────────────────────────────────────────
    tracker.start("Save Results");
    try {
        for (auto& r : results) save_result(r, perf_dir);
        tracker.done("Save Results");
    } catch (const std::exception& e) {
        tracker.fail("Save Results");
        ui::error(e.what());
        return 1;
    }

    tracker.summary();
    for (auto& r : results) print_summary(r);

    return 0;
}
