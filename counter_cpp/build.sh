#!/bin/bash
# Build script for bag_counter C++ project
# Usage:
#   ./build.sh                    # ONNX only (default)
#   ./build.sh -DRKNN=ON          # RKNN on RK3588
#   ./build.sh -DONNX=OFF         # No ONNX
#   ./build.sh -DCLEAN=ON         # Clean build

BUILD_DIR="build"
CLEAN="OFF"
WITH_ONNX="ON"
WITH_RKNN="OFF"

for arg in "$@"; do
  case $arg in
    -DCLEAN=ON) CLEAN="ON" ;;
    -DONNX=OFF) WITH_ONNX="OFF" ;;
    -DRKNN=ON) WITH_RKNN="ON" ;;
    *)
      echo "Unknown option: $arg"
      exit 1
      ;;
  esac
done

if [ "$CLEAN" = "ON" ]; then
  rm -rf "$BUILD_DIR"
fi

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

cmake .. -DWITH_ONNX="$WITH_ONNX" -DWITH_RKNN="$WITH_RKNN" -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

if [ $? -eq 0 ]; then
  echo ""
  echo "Build successful!"
  echo "Run: ./build/bag_counter --source video.mp4 --show"
  echo "     ./build/bag_counter --source video.mp4 --save --show"
  echo "     ./build/bag_counter --config config.yaml --source 0"
fi
