cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 14)

project(SuperVoxel_Cluster)

add_subdirectory(third_party/pybind11)
include_directories(third_party/eigen)


find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(PCL REQUIRED
             common
             features
             filters
             kdtree
             io
             registration
             search
             sample_consensus
             segmentation
             surface
             )


pybind11_add_module(SuperVoxel_Cluster SuperVoxel_Cluster.cpp)


target_link_libraries(SuperVoxel_Cluster PRIVATE ${OpenCV_LIBS})



link_directories(${PCL_LIBRARY_DIRS})

include_directories(${PCL_INCLUDE_DIRS})

target_link_libraries(SuperVoxel_Cluster PRIVATE ${PCL_LIBRARIES})