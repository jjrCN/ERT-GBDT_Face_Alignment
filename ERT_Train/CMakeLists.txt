set(source
    ./code/src/main.cpp
    )

add_executable(ERT_Train ${source})
add_dependencies(ERT_Train ERT)

target_include_directories(ERT_Train PUBLIC
    ${OpenCV_INCLUDE_DIRS}
    ${DEPENDENCY_INCLUDE_DIR}
    ../ERT/code/inc
    )

target_link_libraries(ERT_Train
    ${OpenCV_LIBS}
    ERT
    )
