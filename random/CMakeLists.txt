cmake_minimum_required(VERSION 2.6)
project(randomselection)
set(PROJECT_NAME randomselection)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# Boost
set(Boost_USE_STATIC_LIBS   ON)
set(Boost_USE_MULTITHREADED ON)
find_package(Boost COMPONENTS python REQUIRED)

find_package(OpenCV REQUIRED)
find_package(PythonLibs REQUIRED)
find_package(NumPy REQUIRED)
find_package(Objectness REQUIRED)

set(CMAKE_CXX_FLAGS "-std=c++11 -g")

include_directories(
	include
	${PYTHON_INCLUDE_DIRS}
	${NUMPY_INCLUDE_DIRS}
)

add_library(random_selection MODULE src/main.cpp src/conversion.cpp)
set_target_properties(random_selection PROPERTIES PREFIX "")
target_link_libraries(random_selection
	${Boost_LIBRARIES}
	${OpenCV_LIBRARIES}
	${PYTHON_LIBRARIES}
	${OBJECTNESS_LIBRARIES}
)

add_executable(random src/main.cpp src/conversion.cpp)
target_link_libraries(random
	${Boost_LIBRARIES}
	${OpenCV_LIBRARIES}
	${PYTHON_LIBRARIES}
	${OBJECTNESS_LIBRARIES}
)

add_library(level_selection MODULE src/levels.cpp src/conversion.cpp)
set_target_properties(level_selection PROPERTIES PREFIX "")
target_link_libraries(level_selection
	${Boost_LIBRARIES}
	${OpenCV_LIBRARIES}
	${PYTHON_LIBRARIES}
)
