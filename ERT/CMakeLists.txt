set(source
    ./code/src/ERT.cpp
    ./code/src/regressor.cpp
    ./code/src/tree.cpp
    ./code/src/utils.cpp
    )

add_library(ERT STATIC ${source})
add_dependencies(ERT ${DEPENDENCY_LIST})

target_include_directories(ERT PUBLIC
    ${OpenCV_INCLUDE_DIRS}
    ${DEPENDENCY_INCLUDE_DIR}
    ./code/inc
    )

# target_compile_options(ERT_Train PRIVATE ${OpenMP_CXX_FLAGS})