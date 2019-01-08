cmake_minimum_required(VERSION 2.8)
project(test_svm)

## Add support for C++11, supported in ROS Kinetic and newer
add_definitions(-std=c++11)

find_package(OpenCV)

add_executable(test_svm 
               src/detect_lane_change.cpp
               )
target_link_libraries(test_svm 
                      ${OpenCV_LIBRARIES}
                      dl
                      )
                
