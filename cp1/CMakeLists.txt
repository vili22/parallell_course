SET(target_name "cp1")

add_executable(${target_name} Testcp1.cpp cp.cc)

target_link_libraries(${target_name} 
    PRIVATE 
        ${OpenCV_LIBS}
        OpenMP::OpenMP_CXX)