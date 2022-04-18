cmake_minimum_required(VERSION 3.16) 
project(hw12) 
find_package(osqp REQUIRED)


set(CMAKE_CXX_STANDARD 17) 
add_compile_options(-g)
add_executable(q3_4 q3_4.cpp) 
target_link_libraries(q3_4 osqp::osqp)
target_link_libraries(q3_4 armadillo)