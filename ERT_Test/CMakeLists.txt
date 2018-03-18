project(ERT_Test)
cmake_minimum_required(VERSION 2.8.12)

find_package(OpenCV REQUIRED)

include_directories(/usr/include /usr/local/include)

set(source ./code/src/main.cpp)

add_executable(ERT_Test ${source})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

target_link_libraries(ERT_Test ${OpenCV_LIBS} /usr/lib/x86_64-linux-gnu/libboost_system.so)
