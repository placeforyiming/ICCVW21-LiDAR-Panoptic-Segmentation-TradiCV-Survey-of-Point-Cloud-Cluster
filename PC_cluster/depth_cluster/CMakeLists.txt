cmake_minimum_required(VERSION 3.10)

project(Depth_Cluster)

add_subdirectory(third_party/pybind11)

pybind11_add_module(Depth_Cluster Depth_Cluster.cpp)
