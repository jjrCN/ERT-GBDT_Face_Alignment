project(ERT_Train)
cmake_minimum_required(VERSION 2.8.12)

find_package(OpenCV REQUIRED)

include_directories(/usr/include /usr/local/include ./code/inc)

set(source ./code/src/main.cpp ./code/src/ERT.cpp ./code/src/regressor.cpp ./code/src/tree.cpp ./code/src/utilis.cpp)

add_executable(ERT_Train ${source})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

target_link_libraries(ERT_Train ${OpenCV_LIBS} /usr/lib/x86_64-linux-gnu/libboost_system.so)
