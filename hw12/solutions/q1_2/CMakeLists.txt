cmake_minimum_required(VERSION 3.16) 
project(hw12) 

set(CMAKE_CXX_STANDARD 17) 
add_compile_options(-O3)
add_executable(q1_2 q1_2.cpp) 
target_link_libraries(q1_2 armadillo)
