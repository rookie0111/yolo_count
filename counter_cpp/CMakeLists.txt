cmake_minimum_required(VERSION 3.14)
project(bag_counter LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找包
find_package(OpenCV REQUIRED)
find_package(CUDA QUIET)

# ONNX Runtime
set(ONNXRUNTIME_ROOT "/home/finn/onnxruntime-linux-x64-gpu-1.21.0" CACHE PATH "ONNX Runtime root")

# Eigen3
find_package(Eigen3 QUIET)
if(NOT EIGEN3_INCLUDE_DIR)
    find_path(EIGEN3_INCLUDE_DIR Eigen/Dense
        PATHS /usr/include/eigen3 /usr/local/include/eigen3 ${CMAKE_SOURCE_DIR}/eigen3)
endif()
if(EIGEN3_INCLUDE_DIR)
    add_definitions(-DEIGEN_ENABLE)
    message(STATUS "Eigen3 found: ${EIGEN3_INCLUDE_DIR}")
else()
    message(FATAL_ERROR "Eigen3 not found. Install: sudo apt install libeigen3-dev")
endif()

# RKNN Runtime (仅 RK3588)
option(WITH_RKNN "Build with RKNN support (RK3588)" OFF)
if(WITH_RKNN)
    find_path(RKNN_INCLUDE_DIR rknn_api.h
        PATHS /usr/include /usr/local/include)
    find_library(RKNN_LIB rknnrt
        PATHS /usr/lib /usr/local/lib)
    if(RKNN_INCLUDE_DIR AND RKNN_LIB)
        add_definitions(-DRKNN_ENABLE)
        message(STATUS "RKNN Runtime: ${RKNN_LIB}")
    else()
        message(WARNING "RKNN Runtime not found")
    endif()
endif()

# 包含目录
include_directories(
    ${CMAKE_SOURCE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${ONNXRUNTIME_ROOT}/include
)
if(EIGEN3_INCLUDE_DIR)
    include_directories(${EIGEN3_INCLUDE_DIR})
endif()
if(RKNN_INCLUDE_DIR)
    include_directories(${RKNN_INCLUDE_DIR})
endif()

# 库目录
link_directories(${ONNXRUNTIME_ROOT}/lib)

# 源文件
file(GLOB SOURCES
    ${CMAKE_SOURCE_DIR}/*.cpp
)

# 可执行文件
add_executable(${PROJECT_NAME} ${SOURCES})

# 链接库
target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS}
    onnxruntime
    ${RKNN_LIB}
    pthread
    dl
)

# CUDA 路径（如果可用）
if(CUDA_FOUND)
    set(CUDA_LIB_DIR "/usr/local/cuda-12.4/targets/x86_64-linux/lib")
    link_directories(${CUDA_LIB_DIR})

    set_target_properties(${PROJECT_NAME} PROPERTIES
        BUILD_RPATH "${CUDA_LIB_DIR}:${ONNXRUNTIME_ROOT}/lib"
        INSTALL_RPATH "${CUDA_LIB_DIR}:${ONNXRUNTIME_ROOT}/lib"
    )
endif()

# 编译选项
target_compile_options(${PROJECT_NAME} PRIVATE -Wall -O2)

# 启用 ONNX
add_definitions(-DONNX_ENABLE)

message(STATUS "ONNX Runtime: ${ONNXRUNTIME_ROOT}")
message(STATUS "OpenCV: ${OpenCV_INCLUDE_DIRS}")
# # ONNX 推理
#./build/bag_counter --model /home/finn/ultralytics-main-count/runs/detect/bag_data6/weights/best.onnx --source /mnt/d/download/vidu-video-3321389545428324.mp4 --show --save
./build/bag_counter --model ../runs/detect/bag_data6/weights/best.onnx --source /mnt/d/download/vidu-video-3321389545428324.mp4 --show --save
# 用配置文件
#./build/bag_counter --config config.yaml --source 0 --show

# RKNN 部署 (需加 -DRKNN=ON 重编译)
#cmake -B build -DWITH_RKNN=ON && cmake --build build -j4
#./build/bag_counter --model bag_counter.rknn --backend rknn --source 0 --save