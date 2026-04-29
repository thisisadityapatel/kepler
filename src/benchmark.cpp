#include "benchmark.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <sys/sysctl.h>
#include <sys/utsname.h>
#include <curl/curl.h>

// ── helpers ────────────────────────────────────────────────────────────────

static std::string now_iso8601() {
    auto now   = std::chrono::system_clock::now();
    auto now_t = std::chrono::system_clock::to_time_t(now);
    auto us    = std::chrono::duration_cast<std::chrono::microseconds>(
                     now.time_since_epoch()) % 1'000'000;
    std::ostringstream oss;
    oss << std::put_time(std::localtime(&now_t), "%Y-%m-%dT%H:%M:%S");
    oss << "." << std::setfill('0') << std::setw(6) << us.count();
    return oss.str();
}

static double vec_median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

static size_t write_cb(void* ptr, size_t size, size_t nmemb, std::string* s) {
    s->append(static_cast<char*>(ptr), size * nmemb);
    return size * nmemb;
}

static std::string http_post_json(const std::string& url, const std::string& body) {
    CURL* curl = curl_easy_init();
    if (!curl) throw std::runtime_error("curl_easy_init failed");

    std::string response;
    curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, "Content-Type: application/json");

    curl_easy_setopt(curl, CURLOPT_URL,           url.c_str());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS,    body.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER,    headers);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_cb);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA,     &response);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT,       300L);

    CURLcode res = curl_easy_perform(curl);
    long code = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &code);
    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);

    if (res != CURLE_OK)
        throw std::runtime_error(std::string("curl: ") + curl_easy_strerror(res));
    if (code != 200)
        throw std::runtime_error("HTTP " + std::to_string(code));
    return response;
}

// ── SystemInfo ─────────────────────────────────────────────────────────────

SystemInfo get_system_info() {
    SystemInfo info;
    struct utsname u{};
    uname(&u);
    info.platform     = u.sysname;
    info.architecture = u.machine;

    char buf[256]{};
    size_t sz = sizeof(buf);
    if (sysctlbyname("machdep.cpu.brand_string", buf, &sz, nullptr, 0) == 0)
        info.processor = buf;
    else
        info.processor = info.architecture;

    return info;
}

// ── single inference call ──────────────────────────────────────────────────

BenchmarkIteration run_once(const BenchmarkConfig& cfg) {
    std::string url = "http://" + cfg.host + ":" + std::to_string(cfg.port) + "/completion";

    nlohmann::json payload = {
        {"prompt",      cfg.prompt},
        {"n_predict",   cfg.max_tokens},
        {"temperature", cfg.temperature},
    };

    auto t0       = std::chrono::steady_clock::now();
    auto body     = http_post_json(url, payload.dump());
    auto t1       = std::chrono::steady_clock::now();
    double wall_s = std::chrono::duration<double>(t1 - t0).count();

    auto j       = nlohmann::json::parse(body);
    auto timings = j.value("timings", nlohmann::json::object());

    BenchmarkIteration it;
    it.wall_s               = wall_s;
    it.output_text          = j.value("content", "");
    it.prompt_tokens        = j.value("tokens_evaluated", 0);
    it.ttft_ms              = timings.value("prompt_ms", 0.0);
    it.prefill_ms           = it.ttft_ms;
    it.generated_tokens     = timings.value("predicted_n", 0);
    it.generation_ms        = timings.value("predicted_ms", 0.0);
    it.generation_tok_per_s = timings.value("predicted_per_second", 0.0);
    it.tok_per_s            = wall_s > 0 ? it.generated_tokens / wall_s : 0.0;

    return it;
}

// ── full benchmark run ─────────────────────────────────────────────────────

BenchmarkResult run_benchmark(const std::string& model_path, const BenchmarkConfig& cfg) {
    std::vector<BenchmarkIteration> iters;

    for (int i = 0; i < cfg.iterations; i++) {
        try {
            auto it = run_once(cfg);
            std::cout << "  [" << (i + 1) << "/" << cfg.iterations << "] "
                      << std::fixed << std::setprecision(1)
                      << it.tok_per_s << " tok/s  "
                      << it.generated_tokens << " tokens  "
                      << it.wall_s << "s\n";
            iters.push_back(it);
        } catch (const std::exception& e) {
            std::cerr << "  [" << (i + 1) << "/" << cfg.iterations
                      << "] failed: " << e.what() << "\n";
        }
    }

    if (iters.empty())
        throw std::runtime_error("All benchmark iterations failed");

    std::vector<double> walls, toks, ttfts, gen_toks;
    for (auto& it : iters) {
        walls.push_back(it.wall_s);
        toks.push_back(it.tok_per_s);
        ttfts.push_back(it.ttft_ms);
        gen_toks.push_back(it.generation_tok_per_s);
    }

    BenchmarkSummary summary;
    summary.median_wall_s               = vec_median(walls);
    summary.median_tok_per_s            = vec_median(toks);
    summary.median_ttft_ms              = vec_median(ttfts);
    summary.median_generation_tok_per_s = vec_median(gen_toks);

    std::string stem    = std::filesystem::path(model_path).stem().string();
    std::string repo_id = "local/" + stem;

    BenchmarkResult result;
    result.timestamp   = now_iso8601();
    result.repo_id     = repo_id;
    result.model_ref   = model_path;
    result.system_info = get_system_info();
    result.config      = cfg;
    result.iterations  = iters;
    result.summary     = summary;

    return result;
}

// ── JSON output ────────────────────────────────────────────────────────────

void save_result(const BenchmarkResult& r, const std::filesystem::path& perf_dir) {
    std::filesystem::create_directories(perf_dir);

    std::string date = r.timestamp.substr(0, 10);

    auto clean = [](std::string s, char from, char to) {
        for (char& c : s) if (c == from) c = to;
        return s;
    };
    std::string repo_clean   = clean(clean(r.repo_id, '/', '_'), '.', '_');
    std::string engine_clean = clean(r.engine, '-', '_');
    std::string filename     = date + "_" + repo_clean + "_" + engine_clean + "_" + r.mode + ".json";

    nlohmann::json j;
    j["timestamp"]   = r.timestamp;
    j["repo_id"]     = r.repo_id;
    j["model_ref"]   = r.model_ref;
    j["engine"]      = r.engine;
    j["system_info"] = {
        {"platform",     r.system_info.platform},
        {"architecture", r.system_info.architecture},
        {"processor",    r.system_info.processor},
    };
    j["config"] = {
        {"prompt_set",  r.config.prompt_set},
        {"prompt",      r.config.prompt},
        {"max_tokens",  r.config.max_tokens},
        {"temperature", r.config.temperature},
        {"iterations",  r.config.iterations},
        {"host",        r.config.host},
        {"port",        r.config.port},
    };

    auto iters = nlohmann::json::array();
    for (auto& it : r.iterations) {
        iters.push_back({
            {"wall_s",               it.wall_s},
            {"output_text",          it.output_text},
            {"prompt_tokens",        it.prompt_tokens},
            {"generated_tokens",     it.generated_tokens},
            {"tok_per_s",            it.tok_per_s},
            {"generation_tok_per_s", it.generation_tok_per_s},
            {"ttft_ms",              it.ttft_ms},
            {"prefill_ms",           it.prefill_ms},
            {"generation_ms",        it.generation_ms},
        });
    }
    j["iterations"] = iters;
    j["summary"] = {
        {"median_wall_s",               r.summary.median_wall_s},
        {"median_tok_per_s",            r.summary.median_tok_per_s},
        {"median_ttft_ms",              r.summary.median_ttft_ms},
        {"median_generation_tok_per_s", r.summary.median_generation_tok_per_s},
    };
    j["mode"]        = r.mode;
    j["environment"] = r.environment;

    auto path = perf_dir / filename;
    std::ofstream f(path);
    f << j.dump(2) << "\n";

    std::cout << "[kepler] Results saved: " << path.string() << "\n";
}

// ── console summary ────────────────────────────────────────────────────────

void print_summary(const BenchmarkResult& r) {
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "BENCHMARK RESULTS: " << r.repo_id << "\n";
    std::cout << std::string(60, '=') << "\n";
    std::cout << "Model:      " << r.model_ref << "\n";
    std::cout << "Preset:     " << r.config.prompt_set << "\n";
    std::cout << "Iterations: " << r.iterations.size() << "\n";
    std::cout << std::fixed;
    std::cout << "\nMedian performance:\n";
    std::cout << "  tok/s (overall):  " << std::setprecision(2) << r.summary.median_tok_per_s << "\n";
    std::cout << "  tok/s (gen only): " << std::setprecision(2) << r.summary.median_generation_tok_per_s << "\n";
    std::cout << "  TTFT:             " << std::setprecision(1) << r.summary.median_ttft_ms << " ms\n";
    std::cout << "  Wall time:        " << std::setprecision(2) << r.summary.median_wall_s << " s\n";
    std::cout << "\nSystem: " << r.system_info.platform << " "
              << r.system_info.architecture << " — " << r.system_info.processor << "\n";
    std::cout << std::string(60, '=') << "\n";
}
