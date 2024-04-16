SET(target_name "mf")

add_executable(${target_name} TestMf.cpp mf.cc)

target_link_libraries(${target_name} PUBLIC ${OpenCV_LIBS})