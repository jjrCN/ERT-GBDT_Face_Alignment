
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

ExternalProject_Add(
    dep-eigen
    URL https://gitlab.com/libeigen/eigen/-/archive/3.3.8/eigen-3.3.8.tar.bz2
    DOWNLOAD_DIR ${PROJECT_BINARY_DIR}/downloads
    UPDATE_COMMAND ""
    PATCH_COMMAND ""
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND
        ${CMAKE_COMMAND} -E remove_directory ${DEPENDENCY_INSTALL_DIR}/include/Eigen &&
        ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_BINARY_DIR}/dep-eigen-prefix/src/dep-eigen/Eigen
            ${DEPENDENCY_INSTALL_DIR}/include/Eigen
    TEST_COMMAND ""
    )
set(DEPENDENCY_LIST ${DEPENDENCY_LIST} dep-eigen)

ExternalProject_Add(
    dep-clipp
    GIT_REPOSITORY "https://github.com/muellan/clipp.git"
    GIT_TAG "v1.2.2"
    GIT_SHALLOW 1
    UPDATE_COMMAND ""
    PATCH_COMMAND ""
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND
        ${CMAKE_COMMAND} -E make_directory ${DEPENDENCY_INSTALL_DIR}/include &&
        ${CMAKE_COMMAND} -E copy
            ${CMAKE_CURRENT_BINARY_DIR}/dep-clipp-prefix/src/dep-clipp/include/clipp.h
            ${DEPENDENCY_INSTALL_DIR}/include
    TEST_COMMAND ""
    )
set(DEPENDENCY_LIST ${DEPENDENCY_LIST} dep-clipp)