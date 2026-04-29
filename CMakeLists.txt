cmake_minimum_required(VERSION 3.20)
project(kepler CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(yaml-cpp      REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(CURL          REQUIRED)

# CLI11 is header-only; find_path is more reliable than find_package in Nix
find_path(CLI11_INCLUDE_DIR
  NAMES CLI11.hpp CLI/CLI.hpp
  REQUIRED)
message(STATUS "CLI11 headers: ${CLI11_INCLUDE_DIR}")

add_executable(kepler
  src/main.cpp
  src/ui.cpp
  src/config.cpp
  src/models.cpp
  src/server.cpp
  src/benchmark.cpp
)

include_directories(${CLI11_INCLUDE_DIR} ${CURL_INCLUDE_DIRS})

target_include_directories(kepler PRIVATE src)

target_link_libraries(kepler PRIVATE
  yaml-cpp
  nlohmann_json::nlohmann_json
  CURL::libcurl
)
