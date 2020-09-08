cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
project(sfm LANGUAGES CXX)

set (src_files main.cpp ReadImgData.cpp PnP.cpp FeatureMatcherSURF.cpp)

find_package(OpenCV REQUIRED PATHS /home/spandana/opencv/lib/cmake)

add_executable(sfm ${src_files})

target_link_libraries(sfm ${OpenCV_LIBS})
