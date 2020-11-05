set(source
    ./code/src/main.cpp
    )
    
add_executable(ERT_Test ${source})

target_include_directories(ERT_Test PUBLIC
    ${OpenCV_DIR}/include
    ${DEPENDENCY_INCLUDE_DIR}
    ./code/inc
    )

target_link_libraries(ERT_Test
    ${OpenCV_LIBS}
    )
