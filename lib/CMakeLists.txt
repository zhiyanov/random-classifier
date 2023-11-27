cmake_minimum_required(VERSION 3.10)
cmake_policy(VERSION 3.10)
set(CMAKE_CXX_STANDARD 14)

project(fast)

# add_compile_options(-O0)
# add_compile_options(-Wall)
# add_compile_options(-march=native)

add_compile_options(-g)
add_compile_options(-glldb)

include_directories(eigen)
add_subdirectory(pybind11)

pybind11_add_module(
 	fast SHARED
	lib.cpp
)

