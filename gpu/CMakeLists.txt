cmake_minimum_required(VERSION 2.6)
project(randomselection)
set(PROJECT_NAME randomselection)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

set(CMAKE_CXX_COMPILER ${CUDA_NVCC_EXECUTABLE})
#set(CMAKE_CXX_LINK_FLAGS "")
#set(CMAKE_EXECUTABLE_RUNTIME_CXX_FLAG "")
set(CMAKE_SHARED_LIBRARY_CXX_FLAGS "")
set(CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS "")
set(CMAKE_SHARED_LIBRARY_SONAME_CXX_FLAG "")

#find_package(OpenCV REQUIRED)
find_package(OpenCVlibstdcxx REQUIRED)
find_package(CUDA REQUIRED)
find_package(Thrust REQUIRED)

find_package(PythonLibs REQUIRED)
find_package(NumPy REQUIRED)

# Boost
set(Boost_USE_STATIC_LIBS   ON)
set(Boost_USE_MULTITHREADED ON)
find_package(Boost COMPONENTS python REQUIRED)

set(CUDA_LIBRARIES "")

list(APPEND CUDA_NVCC_FLAGS -ccbin clang)
list(APPEND CUDA_NVCC_FLAGS -Xcompiler -arch)
list(APPEND CUDA_NVCC_FLAGS -Xcompiler x86_64)
list(APPEND CUDA_NVCC_FLAGS -Xcompiler -fPIC)
list(APPEND CUDA_NVCC_FLAGS -Xcompiler -dynamiclib)
list(APPEND CUDA_NVCC_FLAGS -Xcompiler -install_name)
list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_20,code=sm_20)
list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_30,code=sm_30)
list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_35,code=sm_35)
list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_37,code=sm_37)
list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_50,code=sm_50)
list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_50,code=compute_50)

#set(CMAKE_CXX_FLAGS "-std=c++11 -O3")
#set(CMAKE_CXX_FLAGS "-ferror-limit=20")

include_directories(
	include
	${PYTHON_INCLUDE_DIRS}
	${NUMPY_INCLUDE_DIRS}
	${CUDA_INCLUDE_DIRS}
	${THRUST_INCLUDE_DIRS}
)

cuda_compile(SEGMENTATION_O segmentationTree.cu)
cuda_add_library(segmentation ${SEGMENTATION_O})
target_link_libraries(segmentation ${OPENCV_LIBSTDCXX_LIBRARIES})

#cuda_add_executable(segment_test ${SEGMENTATION_O})
#target_link_libraries(segment_test ${OPENCV_LIBSTDCXX_LIBRARIES})

cuda_add_library(gpu_segmentation SHARED main.cpp conversion.cpp)
set_target_properties(gpu_segmentation PROPERTIES PREFIX "")
target_link_libraries(gpu_segmentation
	segmentation
	${Boost_LIBRARIES}
	${OpenCV_LIBRARIES}
	${PYTHON_LIBRARIES}
)
