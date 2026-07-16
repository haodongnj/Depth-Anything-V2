add_library(depth_trt STATIC
    src/engine.cpp
    src/preprocess.cpp
    src/visualize.cpp
)

target_include_directories(depth_trt
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(depth_trt
    PUBLIC
        TensorRT::nvinfer
        CUDA::cudart
        opencv_core
        opencv_imgproc
)
