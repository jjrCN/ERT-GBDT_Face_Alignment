project(ERT_Train)
cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 11)

include(Dependency.cmake)

# set(OpenCV_DIR  ${CMAKE_CURRENT_SOURCE_DIR}/../opencv/install-debug)
set(OpenCV_DIR  ${CMAKE_CURRENT_SOURCE_DIR}/../opencv/install)

find_package(OpenCV REQUIRED core highgui imgproc objdetect)
message("opencv dir: ${OpenCV_INCLUDE_DIRS}")
message("opencv libs: ${OpenCV_LIBS}")

# find_package(OpenMP REQUIRED)
# message("openmp compile option: ${OpenMP_CXX_FLAGS}")

add_subdirectory(ERT)
add_subdirectory(ERT_Train)
add_subdirectory(ERT_Test)