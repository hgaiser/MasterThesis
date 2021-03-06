set(INC_DIRS ${BING_SOURCE_DIR}/LibLinear ${BING_SOURCE_DIR}/LibLinear/blas)
set(LIB_DIRS ${BING_BINARY_DIR}/LibLinear)

find_package(OpenCV REQUIRED)

#OPENCV_DIR should be set as an environment variable in the shell if it is not
#present in the standard directories
if (DEFINED ENV{OPENCV_DIR})
  list(APPEND INC_DIRS ${OPENCV_DIR}/include)
  list(APPEND LIB_DIRS ${OPENCV_DIR}/lib)
endif (DEFINED ENV{OPENCV_DIR})

include_directories(${OPENCV_INCLUDE_DIRS} ${INC_DIRS})
link_directories(${LIB_DIRS})

if (${CMAKE_COMPILER_IS_GNUCXX})
  set(OPENMP_FLAG "-fopenmp -std=c++11 -ftree-vectorize")
endif (${CMAKE_COMPILER_IS_GNUCXX})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OPENMP_FLAG} -O3")

#add_executable(${PROJECT_NAME} Main.cpp CmFile.cpp CmShow.cpp DataSetVOC.cpp FilterTIG.cpp Objectness.cpp stdafx.cpp)
#target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBRARIES} LIBLINEAR BLAS)

add_library(objectness SHARED CmFile.cpp CmShow.cpp DataSetVOC.cpp FilterTIG.cpp Objectness.cpp)
target_link_libraries(objectness ${OpenCV_LIBRARIES} LIBLINEAR BLAS)
