add_executable(depth_image main.cpp)
target_link_libraries(depth_image PRIVATE depth_trt opencv_imgcodecs)
