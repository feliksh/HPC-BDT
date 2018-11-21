cmake_minimum_required(VERSION 3.8)
project(HPC-BDT)

set(CMAKE_CXX_STANDARD 14)

SET(CMAKE_CXX_FLAGS -pthread)

set(SOURCE_FILES
        main.cc)

add_executable(hpc-bdt ${SOURCE_FILES})