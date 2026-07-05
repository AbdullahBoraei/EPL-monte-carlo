cmake_minimum_required(VERSION 3.16)
project(epl_monte_carlo CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# A Monte Carlo engine benchmarked at -O0 measures nothing: unoptimized
# builds are 10-50x slower and misrepresent every design decision.
# Default to Release (-O3) unless the caller explicitly asks otherwise.
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# The engine: a static library. The CLI and (later) tests and benchmarks
# all link against this one target -- the code is written once and every
# executable is a thin consumer.
add_library(simulator STATIC
  src/csv.cpp
  src/season.cpp
  src/engine.cpp
)
target_include_directories(simulator PUBLIC include)
target_compile_options(simulator PRIVATE -Wall -Wextra)

add_executable(simulate cli/main.cpp)
target_link_libraries(simulate PRIVATE simulator)
target_compile_options(simulate PRIVATE -Wall -Wextra)

enable_testing()
add_executable(tests tests/test_main.cpp)
target_link_libraries(tests PRIVATE simulator)
target_compile_options(tests PRIVATE -Wall -Wextra)
add_test(NAME unit COMMAND tests)
