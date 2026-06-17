# Bag Counter C++ 项目结构
#
# 编译依赖:
#   sudo apt install libopencv-dev libeigen3-dev
#
#   # ONNX Runtime (可选):
#   wget https://github.com/microsoft/onnxruntime/releases/.../onnxruntime-linux-aarch64-1.x.x.tgz
#   tar xzf onnxruntime-linux-aarch64-*.tgz -C /usr/local --strip-components=1
#
#   # RKNN (仅 RK3588):
#   # 从 Firefly/Rockchip SDK 获取 librknnrt.so 和 rknn_api.h
#
# 编译:
#   chmod +x build.sh && ./build.sh
#
# 使用:
#   ./build/bag_counter --source video.mp4 --show --save

SOURCES = main.cpp yolo_detector.cpp bytetrack.cpp counter.cpp
EXECUTABLE = bag_counter

CXX = g++
CXXFLAGS = -std=c++17 -O2 -Wall

# OpenCV
OPENCV_FLAGS = $(shell pkg-config --cflags --libs opencv4 2>/dev/null || pkg-config --cflags --libs opencv)

# Eigen3
EIGEN_FLAGS = -I/usr/include/eigen3

# ONNX Runtime (optional, uncomment to enable)
# ONNX_FLAGS = -DONNX_ENABLE -I/usr/local/include/onnxruntime -L/usr/local/lib -lonnxruntime

all:
	$(CXX) $(CXXFLAGS) $(SOURCES) -o $(EXECUTABLE) $(OPENCV_FLAGS) $(EIGEN_FLAGS) $(ONNX_FLAGS)

clean:
	rm -f $(EXECUTABLE)
