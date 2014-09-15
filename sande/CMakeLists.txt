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

set(CMAKE_CXX_FLAGS "-std=c++11 -O3")

include_directories(
	${PYTHON_INCLUDE_DIRS}
	${NUMPY_INCLUDE_DIRS}
)

add_library(selective_search MODULE src/main.cpp src/conversion.cpp)
set_target_properties(selective_search PROPERTIES PREFIX "")
target_link_libraries(selective_search
	${Boost_LIBRARIES}
	${OpenCV_LIBRARIES}
	${PYTHON_LIBRARIES}
)

add_executable(test src/main.cpp src/conversion.cpp)
target_link_libraries(test
	${Boost_LIBRARIES}
	${OpenCV_LIBRARIES}
	${PYTHON_LIBRARIES}
)
