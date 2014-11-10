cmake_minimum_required(VERSION 2.6)
project(gpb)
set(PROJECT_NAME gpb)

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
	include
	${PYTHON_INCLUDE_DIRS}
	${NUMPY_INCLUDE_DIRS}
)

add_library(gpb_selection MODULE src/main.cpp src/conversion.cpp)
set_target_properties(gpb_selection PROPERTIES PREFIX "")
target_link_libraries(gpb_selection
	${Boost_LIBRARIES}
	${OpenCV_LIBRARIES}
	${PYTHON_LIBRARIES}
)

add_executable(gpb src/main.cpp src/conversion.cpp)
target_link_libraries(gpb
	${Boost_LIBRARIES}
	${OpenCV_LIBRARIES}
	${PYTHON_LIBRARIES}
)
