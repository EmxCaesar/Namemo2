cmake_minimum_required(VERSION 3.8)
project(webserver)

############################---FLAG---#########################

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "-std=c++14 -fPIC -Wall -g -Ofast -Wfatal-errors -fopenmp -pthread -lmysqlclient" )
add_definitions(-D_MWAITXINTRIN_H_INCLUDED -DWITH_OPENSSL  -DWITH_DOM)

###########################---LIB---###########################
find_package( OpenCV REQUIRED )
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(Eigen3 REQUIRED)
include_directories(${Eigen3_INCLUDE_DIRS})

find_package(OpenSSL REQUIRED)
include_directories(${OpenSSL_INCLUDE_DIR})

###########################---PROJ---###########################

add_subdirectory(code)
add_executable(server  ./code/main.cpp)
target_link_libraries(server webserver)
target_link_libraries(server smartclass)


