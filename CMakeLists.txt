cmake_minimum_required(VERSION 3.28)
project(filterer CXX)

find_package(fmt REQUIRED)
find_package(OpenEXR REQUIRED)
find_package(PNG REQUIRED)
find_package(TBB REQUIRED)
find_package(Boost CONFIG REQUIRED COMPONENTS container)

set(CMAKE_CXX_STANDARD 23)

add_library(
  filtlib OBJECT
  src/filter.cpp
  src/io.cpp
  src/mempool.cpp
  src/util.cpp
)
target_link_libraries(
  filtlib PUBLIC
  OpenEXR::OpenEXR
  fmt::fmt
  PNG::PNG
  TBB::tbb
)

add_executable(filter src/main.cpp)
target_link_libraries(filter PRIVATE filtlib)