cmake_minimum_required(VERSION 3.20)
project(gpt2_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)  # generates build/compile_commands.json

# Default to Debug so assertions (bounds checks, shape checks) stay live under
# ctest. Pass -DCMAKE_BUILD_TYPE=Release explicitly for timed forward passes --
# a Debug forward pass over 12 layers is slow enough to be annoying.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Debug CACHE STRING "" FORCE)
endif()

add_compile_options(-Wall -Wextra)

# Core library: tensor + autograd (carried over from mnist-cpp), and everything
# else that lands in src/ -- weight loader, transformer blocks, the model.
file(GLOB CORE_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
add_library(gpt2core STATIC ${CORE_SOURCES})
target_include_directories(gpt2core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(gpt2core PUBLIC cxx_std_23)

# Executables: any apps/*.cpp becomes a binary of the same name.
file(GLOB APP_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/apps/*.cpp)
foreach(app_src ${APP_SOURCES})
  get_filename_component(app_name ${app_src} NAME_WE)
  add_executable(${app_name} ${app_src})
  target_link_libraries(${app_name} PRIVATE gpt2core)
endforeach()

# Tests: any tests/*.cpp becomes a ctest case.
enable_testing()
file(GLOB TEST_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp)
foreach(test_src ${TEST_SOURCES})
  get_filename_component(test_name ${test_src} NAME_WE)
  add_executable(${test_name} ${test_src})
  target_link_libraries(${test_name} PRIVATE gpt2core)
  add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()
