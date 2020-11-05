
set(source
    ./code/src/main.cpp
    ./code/src/ERT.cpp
    ./code/src/regressor.cpp
    ./code/src/tree.cpp
    ./code/src/utilis.cpp
    )

add_executable(ERT_Train ${source})
add_dependencies(ERT_Train ${DEPENDENCY_LIST})

target_include_directories(ERT_Train PUBLIC
    ${OpenCV_DIR}/include
    ${DEPENDENCY_INCLUDE_DIR}
    ./code/inc
    )

target_link_libraries(ERT_Train
    ${OpenCV_LIBS}
    )
