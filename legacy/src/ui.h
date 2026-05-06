#pragma once
#include <chrono>
#include <string>
#include <vector>

namespace ui {

// ── ANSI codes ─────────────────────────────────────────────────────────────
inline constexpr const char* RESET  = "\033[0m";
inline constexpr const char* BOLD   = "\033[1m";
inline constexpr const char* DIM    = "\033[2m";
inline constexpr const char* GREEN  = "\033[32m";
inline constexpr const char* YELLOW = "\033[33m";
inline constexpr const char* RED    = "\033[31m";
inline constexpr const char* CYAN   = "\033[36m";
inline constexpr const char* WHITE  = "\033[97m";

void clear_screen();
void print_banner();

void info   (const std::string& msg);
void success(const std::string& msg);
void warn   (const std::string& msg);
void error  (const std::string& msg);

// Interactive prompts — print below whatever is already on screen
std::string pick_benchmark();   // returns: quick | standard | performance | hard

// ── Workflow tracker ───────────────────────────────────────────────────────

struct Step {
    std::string name;
    enum class Status { pending, running, done, failed } status = Status::pending;
    double elapsed_s = 0.0;
    std::chrono::steady_clock::time_point started;
};

class WorkflowTracker {
public:
    void add(const std::string& name);
    void start(const std::string& name);    // clears screen, redraws, marks running
    void done(const std::string& name);     // marks done, redraws
    void fail(const std::string& name);     // marks failed, redraws
    void redraw() const;
    void summary() const;

private:
    std::vector<Step> steps_;
    Step* find(const std::string& name);
};

} // namespace ui
