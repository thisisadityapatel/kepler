#include "models.h"
#include <iomanip>
#include <iostream>
#include <stdexcept>

std::vector<std::filesystem::path> find_gguf_models(const std::filesystem::path& dir) {
    std::vector<std::filesystem::path> models;
    if (!std::filesystem::exists(dir)) return models;

    for (auto& entry : std::filesystem::recursive_directory_iterator(dir)) {
        if (entry.is_regular_file() && entry.path().extension() == ".gguf")
            models.push_back(entry.path());
    }
    return models;
}

std::filesystem::path select_model_interactive(const std::vector<std::filesystem::path>& models) {
    if (models.empty())
        throw std::runtime_error("No .gguf models found in models/ directory");

    if (models.size() == 1) {
        std::cout << "Using: " << models[0].filename().string() << "\n";
        return models[0];
    }

    std::cout << "\nAvailable models:\n";
    for (size_t i = 0; i < models.size(); i++) {
        auto bytes = std::filesystem::file_size(models[i]);
        float gb   = bytes / (1024.0f * 1024.0f * 1024.0f);
        std::cout << "  [" << (i + 1) << "] "
                  << models[i].filename().string()
                  << "  (" << std::fixed << std::setprecision(1) << gb << " GB)\n";
    }

    int choice = 0;
    while (choice < 1 || choice > static_cast<int>(models.size())) {
        std::cout << "Select model (1-" << models.size() << "): ";
        std::cin >> choice;
    }

    return models[choice - 1];
}
