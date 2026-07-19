cmake_minimum_required(VERSION 3.20)
project(DepthAnythingTRT VERSION 1.0.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Export compile_commands.json for clangd LSP
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Allow the user to override CUDA arch
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89")  # Turing through Ada
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

find_package(CUDAToolkit REQUIRED)
find_package(TensorRT REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs highgui videoio)

add_subdirectory(depth_trt)
add_subdirectory(depth_image)
add_subdirectory(depth_camera)
