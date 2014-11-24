cmake_minimum_required(VERSION 2.6)
project(brute_selection)
set(PROJECT_NAME brute_selection)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# Boost
set(Boost_USE_STATIC_LIBS   ON)
set(Boost_USE_MULTITHREADED ON)
find_package(Boost COMPONENTS python REQUIRED)

find_package(OpenCV REQUIRED)
find_package(PythonLibs REQUIRED)
find_package(NumPy REQUIRED)

set(CMAKE_CXX_FLAGS "-std=c++11 -O3")

include_directories(
	include
	${PYTHON_INCLUDE_DIRS}
	${NUMPY_INCLUDE_DIRS}
)

add_library(brute_selection MODULE src/main.cpp src/conversion.cpp)
set_target_properties(brute_selection PROPERTIES PREFIX "")
target_link_libraries(brute_selection
	${Boost_LIBRARIES}
	${OpenCV_LIBRARIES}
	${PYTHON_LIBRARIES}
)

add_executable(brute src/main.cpp src/conversion.cpp)
target_link_libraries(brute
	${Boost_LIBRARIES}
	${OpenCV_LIBRARIES}
	${PYTHON_LIBRARIES}
)
