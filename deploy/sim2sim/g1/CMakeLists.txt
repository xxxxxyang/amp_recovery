cmake_minimum_required(VERSION 3.12)
project(amp_mjlab_g1_sim2sim LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(UNITREE_SDK2_ROOT "/opt/unitree_robotics" CACHE PATH
    "Unitree SDK2 installation prefix")

find_package(unitree_sdk2 CONFIG REQUIRED
  HINTS "${UNITREE_SDK2_ROOT}/lib/cmake/unitree_sdk2")
find_package(Eigen3 CONFIG REQUIRED)
find_package(fmt CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)
find_package(Threads REQUIRED)

find_package(Boost REQUIRED COMPONENTS program_options)
find_package(yaml-cpp REQUIRED)
find_package(ZLIB REQUIRED)

set(ONNXRUNTIME_VERSION "1.22.0")
if(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
  set(ONNXRUNTIME_ARCH "x64")
elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
  set(ONNXRUNTIME_ARCH "aarch64")
else()
  message(FATAL_ERROR "Unsupported CPU architecture: ${CMAKE_SYSTEM_PROCESSOR}")
endif()

set(ONNXRUNTIME_DEFAULT_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/../../thirdparty/onnxruntime-linux-${ONNXRUNTIME_ARCH}-${ONNXRUNTIME_VERSION}")
set(ONNXRUNTIME_ROOT "${ONNXRUNTIME_DEFAULT_DIR}" CACHE PATH
    "Path to the ONNX Runtime ${ONNXRUNTIME_VERSION} distribution")

if(NOT EXISTS "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so.${ONNXRUNTIME_VERSION}")
  message(FATAL_ERROR
    "ONNX Runtime ${ONNXRUNTIME_VERSION} not found under ${ONNXRUNTIME_ROOT}.\n"
    "Run 'bash scripts/setup_sim2sim_deps.sh' from the AMP_mjlab repository root.")
endif()

add_executable(g1_sim2sim_controller
  main.cpp
  State_RLBase.cpp
)
target_include_directories(g1_sim2sim_controller PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}"
  "${CMAKE_CURRENT_SOURCE_DIR}/../../include"
  "${ONNXRUNTIME_ROOT}/include"
)
target_link_libraries(g1_sim2sim_controller PRIVATE
  unitree_sdk2
  Eigen3::Eigen
  fmt::fmt
  spdlog::spdlog
  Threads::Threads
  rt
  Boost::program_options
  yaml-cpp
  "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so.${ONNXRUNTIME_VERSION}"
  ZLIB::ZLIB
)

set_target_properties(g1_sim2sim_controller PROPERTIES
  BUILD_RPATH "${ONNXRUNTIME_ROOT}/lib;${UNITREE_SDK2_ROOT}/lib"
)
