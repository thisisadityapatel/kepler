#include "ui.h"
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace ui {

void clear_screen() {
    std::cout << "\033[2J\033[H" << std::flush;
}

void print_banner() {
    std::cout << CYAN << BOLD
        << "  ✦           *           ✦\n"
        << "      __ __ ______ ____  __    ______ ____\n"
        << "     / //_// ____// __ \\/ /   / ____// __ \\\n"
        << "    / ,<  / __/  / /_/ / /   / __/  / /_/ /\n"
        << "   / /| |/ /___ / ____/ /___/ /___ / _, _/\n"
        << "  /_/ |_|_____//_/   /_____/_____//_/ |_|\n"
        << RESET << "\n"
        << DIM << "  *   model inference and benchmarking   ✦\n" << RESET
        << "\n";
}

// ── inline status lines ────────────────────────────────────────────────────

void info   (const std::string& m) { std::cout << "  " << CYAN   << "●" << RESET << " " << m << "\n"; }
void success(const std::string& m) { std::cout << "  " << GREEN  << "●" << RESET << " " << m << "\n"; }
void warn   (const std::string& m) { std::cout << "  " << YELLOW << "●" << RESET << " " << m << "\n"; }
void error  (const std::string& m) { std::cout << "  " << RED    << "●" << RESET << " " << m << "\n"; }

// ── WorkflowTracker ────────────────────────────────────────────────────────

void WorkflowTracker::add(const std::string& name) {
    steps_.push_back({name});
}

Step* WorkflowTracker::find(const std::string& name) {
    for (auto& s : steps_) if (s.name == name) return &s;
    return nullptr;
}

void WorkflowTracker::redraw() const {
    clear_screen();
    print_banner();

    for (auto& s : steps_) {
        std::string dot, color, timing;

        switch (s.status) {
            case Step::Status::done:
                dot   = "●";
                color = GREEN;
                if (s.elapsed_s > 0) {
                    std::ostringstream ss;
                    ss << std::fixed << std::setprecision(1) << s.elapsed_s << "s";
                    timing = std::string(DIM) + ss.str() + RESET;
                }
                break;
            case Step::Status::running:
                dot   = "●";
                color = YELLOW;
                timing = std::string(YELLOW) + "..." + RESET;
                break;
            case Step::Status::failed:
                dot   = "●";
                color = RED;
                timing = std::string(RED) + "FAILED" + RESET;
                break;
            default:
                dot   = "○";
                color = DIM;
                break;
        }

        // Pad step name to fixed width for alignment
        std::string padded = s.name;
        const int NAME_WIDTH = 20;
        if ((int)padded.size() < NAME_WIDTH)
            padded += std::string(NAME_WIDTH - padded.size(), ' ');

        std::cout << "  " << color << dot << RESET << " "
                  << (s.status == Step::Status::pending ? std::string(DIM) + padded + RESET : padded)
                  << " " << timing << "\n";
    }
    std::cout << "\n";
}

void WorkflowTracker::start(const std::string& name) {
    auto* s = find(name);
    if (!s) return;
    s->status  = Step::Status::running;
    s->started = std::chrono::steady_clock::now();
    redraw();
}

void WorkflowTracker::done(const std::string& name) {
    auto* s = find(name);
    if (!s) return;
    s->status    = Step::Status::done;
    s->elapsed_s = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - s->started).count();
    redraw();
}

void WorkflowTracker::fail(const std::string& name) {
    auto* s = find(name);
    if (!s) return;
    s->status    = Step::Status::failed;
    s->elapsed_s = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - s->started).count();
    redraw();
}

std::string pick_benchmark() {
    std::cout << "  Select benchmark:\n\n"
              << "    " << CYAN << "[1]" << RESET << "  quick         2 iters · 50 tokens\n"
              << "    " << CYAN << "[2]" << RESET << "  standard      3 iters · 100 tokens\n"
              << "    " << CYAN << "[3]" << RESET << "  performance   5 iters · 200 tokens\n"
              << "    " << CYAN << "[4]" << RESET << "  hard          3 questions, 3 iters each\n"
              << "\n";

    int choice = 0;
    while (choice < 1 || choice > 4) {
        std::cout << "  Choice (1-4): ";
        if (!(std::cin >> choice)) { choice = 2; break; }
    }
    switch (choice) {
        case 1:  return "quick";
        case 3:  return "performance";
        case 4:  return "hard";
        default: return "standard";
    }
}

void WorkflowTracker::summary() const {
    int done   = 0, failed = 0;
    double total = 0;
    for (auto& s : steps_) {
        if (s.status == Step::Status::done)   { done++;   total += s.elapsed_s; }
        if (s.status == Step::Status::failed)   failed++;
    }

    std::cout << "\n";
    if (failed == 0 && done == (int)steps_.size()) {
        std::cout << "  " << GREEN << "●" << RESET << " All steps completed in "
                  << std::fixed << std::setprecision(1) << total << "s\n\n";
    } else if (failed > 0) {
        std::cout << "  " << RED << "●" << RESET << " Workflow failed ("
                  << failed << " step" << (failed > 1 ? "s" : "") << ")\n\n";
    }
}

} // namespace ui
