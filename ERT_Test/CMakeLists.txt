set(source
    ./code/src/main.cpp
    )
    
add_executable(ERT_Test ${source})
add_dependencies(ERT_Test ERT)

target_include_directories(ERT_Test PUBLIC
    ${OpenCV_DIR}/include
    ${DEPENDENCY_INCLUDE_DIR}
    ../ERT/code/inc
    )

target_link_libraries(ERT_Test
    ${OpenCV_LIBS}
    ERT
    )
