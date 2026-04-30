#pragma once
#include <filesystem>
#include <vector>

std::vector<std::filesystem::path> find_gguf_models(const std::filesystem::path& dir);
std::filesystem::path select_model_interactive(const std::vector<std::filesystem::path>& models);
