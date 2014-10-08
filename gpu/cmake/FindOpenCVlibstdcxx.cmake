# - Try to find OpenCVlibstdcxx
# Once done this will define
#  OPENCV_LIBSTDCXX_FOUND - System has OpenCVlibstdcxx
#  OPENCV_LIBSTDCXX_INCLUDE_DIRS - The OpenCVlibstdcxx include directories
#  OPENCV_LIBSTDCXX_LIBRARIES - The libraries needed to use OpenCVlibstdcxx
#  OPENCV_LIBSTDCXX_DEFINITIONS - Compiler switches required for using OpenCVlibstdcxx

find_package(PkgConfig)
pkg_check_modules(PC_OPENCV_LIBSTDCXX QUIET opencv-libstdcxx)
set(OPENCV_LIBSTDCXX_DEFINITIONS ${PC_OPENCV_LIBSTDCXX_CFLAGS_OTHER})

find_path(OPENCV_LIBSTDCXX_INCLUDE_DIR opencv2/opencv.hpp
          HINTS ${PC_OPENCV_LIBSTDCXX_INCLUDEDIR} ${PC_OPENCV_LIBSTDCXX_INCLUDE_DIRS})

set(OPENCV_LIBSTDCXX_LIBRARIES ${PC_OPENCV_LIBSTDCXX_LDFLAGS})
set(OPENCV_LIBSTDCXX_LIBRARY ${PC_OPENCV_LIBSTDCXX_LDFLAGS})
set(OPENCV_LIBSTDCXX_INCLUDE_DIRS ${OPENCV_LIBSTDCXX_INCLUDE_DIR} )

include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set OPENCV_LIBSTDCXX_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(OpenCVlibstdcxx  DEFAULT_MSG
                                  OPENCV_LIBSTDCXX_LIBRARY OPENCV_LIBSTDCXX_INCLUDE_DIR)

mark_as_advanced(OPENCV_LIBSTDCXX_INCLUDE_DIR OPENCV_LIBSTDCXX_LIBRARY )
