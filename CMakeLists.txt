cmake_minimum_required(VERSION 3.5)
project(DisplayImage)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(opencv-classifiers main.cpp)
target_link_libraries(opencv-classifiers ${OpenCV_LIBS})
