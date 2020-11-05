project(ERT_Train)
cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 11)

include(ExternalProject)
set(DEPENDENCY_INSTALL_DIR ${PROJECT_BINARY_DIR}/install)
set(DEPENDENCY_INCLUDE_DIR ${DEPENDENCY_INSTALL_DIR}/include)
set(DEPENDENCY_LIBS ${LINK_DIR_OPTION}${DEPENDENCY_INSTALL_DIR}/lib)
ExternalProject_Add(
    dep_filesystem
    GIT_REPOSITORY "https://github.com/gulrak/filesystem.git"
    GIT_TAG "v1.3.2"
    GIT_SHALLOW 1
    UPDATE_COMMAND ""
    PATCH_COMMAND ""
    CMAKE_ARGS
        -DCMAKE_INSTALL_PREFIX=${DEPENDENCY_INSTALL_DIR}
        -DGHC_FILESYSTEM_BUILD_TESTING=OFF
        -DGHC_FILESYSTEM_BUILD_EXAMPLES=OFF
    TEST_COMMAND ""
    )
set(DEPENDENCY_LIST ${DEPENDENCY_LIST} dep_filesystem)

ExternalProject_Add(
    dep-json
    GIT_REPOSITORY "https://github.com/nlohmann/json.git"
    GIT_TAG "v3.9.0"
    GIT_SHALLOW 1
    UPDATE_COMMAND ""
    PATCH_COMMAND ""
    CMAKE_ARGS
        -DCMAKE_INSTALL_PREFIX=${DEPENDENCY_INSTALL_DIR}
        -DJSON_BuildTests=OFF
    TEST_COMMAND ""
    )
set(DEPENDENCY_LIST ${DEPENDENCY_LIST} dep-json)

if (CMAKE_BUILD_TYPE MATCHES DEBUG)
    set(OpenCV_DIR  ${CMAKE_CURRENT_SOURCE_DIR}/../opencv/install-debug)
else()
    set(OpenCV_DIR  ${CMAKE_CURRENT_SOURCE_DIR}/../opencv/install)
endif()

find_package(OpenCV REQUIRED)

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
message("opencv dir: ${OpenCV_DIR}/include")


add_subdirectory(ERT_Train)
add_subdirectory(ERT_Test)