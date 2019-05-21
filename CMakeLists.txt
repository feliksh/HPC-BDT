cmake_minimum_required(VERSION 3.8)
project(HPC-BDT)

set(CMAKE_CXX_STANDARD 14)

SET(CMAKE_CXX_FLAGS "-fopenmp -pthread -march=native -O3")

set(SOURCE_FILES
        main.cc structs.h utility.h bdt.h old_versions.h backfitting.h)

add_executable(hpc-bdt ${SOURCE_FILES})