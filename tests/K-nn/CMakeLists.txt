cmake_minimum_required (VERSION 3.5) # The minimum version of CMake necessary to build this project
project (K-nn) # The name of our project

declare_trisycl_test(TARGET knn_sycl)
if(${TRISYCL_OPENCL})
  declare_trisycl_test(TARGET knn_sycl_opencl USES_OPENCL)
endif(${TRISYCL_OPENCL})
