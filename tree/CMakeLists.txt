cmake_minimum_required(VERSION 2.6)
project(segmentationtree)
set(PROJECT_NAME segmentationtree)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# Boost
set(Boost_USE_STATIC_LIBS   ON)
set(Boost_USE_MULTITHREADED ON)
find_package(Boost COMPONENTS python REQUIRED)

find_package(OpenCV REQUIRED)
find_package(PythonLibs REQUIRED)
find_package(NumPy REQUIRED)

set(CMAKE_CXX_FLAGS "-std=c++11 -g")

include_directories(
	${PYTHON_INCLUDE_DIRS}
	${NUMPY_INCLUDE_DIRS}
)

add_library(segmentation_tree MODULE src/tree.cpp)
set_target_properties(segmentation_tree PROPERTIES PREFIX "")
target_link_libraries(segmentation_tree 
	${Boost_LIBRARIES}
	${OpenCV_LIBRARIES}
	${PYTHON_LIBRARIES}
)

add_executable(seg_tree src/tree.cpp)
target_link_libraries(seg_tree 
	${Boost_LIBRARIES}
	${OpenCV_LIBRARIES}
	${PYTHON_LIBRARIES}
)
