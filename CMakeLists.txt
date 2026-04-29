cmake_minimum_required(VERSION 3.20)
project(kepler CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(yaml-cpp      REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(CLI11         REQUIRED)
find_package(CURL          REQUIRED)

add_executable(kepler
  src/main.cpp
  src/config.cpp
  src/models.cpp
  src/server.cpp
  src/benchmark.cpp
)

target_include_directories(kepler PRIVATE src)
target_link_libraries(kepler PRIVATE
  yaml-cpp
  nlohmann_json::nlohmann_json
  CLI11::CLI11
  CURL::libcurl
)
