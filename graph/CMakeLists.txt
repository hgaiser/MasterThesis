cmake_minimum_required(VERSION 2.6)
project(segmentationtree)
set(PROJECT_NAME segmentation)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

#set(CMAKE_C_COMPILER "gcc-4.9")
#set(CMAKE_CXX_COMPILER "g++-4.9")

# Boost
set(Boost_USE_STATIC_LIBS   ON)
set(Boost_USE_MULTITHREADED ON)
find_package(Boost COMPONENTS python REQUIRED)

find_package(OpenCV REQUIRED)
find_package(PythonLibs REQUIRED)
find_package(NumPy REQUIRED)

set(CMAKE_CXX_FLAGS "-std=c++11 -O3")

include_directories(
	${PYTHON_INCLUDE_DIRS}
	${NUMPY_INCLUDE_DIRS}
)

add_library(segment_felz MODULE src/segment.cpp)
set_target_properties(segment_felz PROPERTIES PREFIX "")
target_link_libraries(segment_felz
	${Boost_LIBRARIES}
	${OpenCV_LIBRARIES}
	${PYTHON_LIBRARIES}
)

add_executable(segment src/segment.cpp)
target_link_libraries(segment
	${Boost_LIBRARIES}
	${OpenCV_LIBRARIES}
	${PYTHON_LIBRARIES}
)
