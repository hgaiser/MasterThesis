# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/Cellar/cmake/2.8.12.2/bin/cmake

# The command to remove a file.
RM = /usr/local/Cellar/cmake/2.8.12.2/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The program to use to edit the cache.
CMAKE_EDIT_COMMAND = /usr/local/Cellar/cmake/2.8.12.2/bin/ccmake

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/hans/Dropbox/Uni/MasterThesis/code/cpp/edge-based

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/hans/Dropbox/Uni/MasterThesis/code/cpp/edge-based/build

# Include any dependencies generated for this target.
include CMakeFiles/seg_tree.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/seg_tree.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/seg_tree.dir/flags.make

CMakeFiles/seg_tree.dir/src/tree.cpp.o: CMakeFiles/seg_tree.dir/flags.make
CMakeFiles/seg_tree.dir/src/tree.cpp.o: ../src/tree.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /Users/hans/Dropbox/Uni/MasterThesis/code/cpp/edge-based/build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/seg_tree.dir/src/tree.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/seg_tree.dir/src/tree.cpp.o -c /Users/hans/Dropbox/Uni/MasterThesis/code/cpp/edge-based/src/tree.cpp

CMakeFiles/seg_tree.dir/src/tree.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/seg_tree.dir/src/tree.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /Users/hans/Dropbox/Uni/MasterThesis/code/cpp/edge-based/src/tree.cpp > CMakeFiles/seg_tree.dir/src/tree.cpp.i

CMakeFiles/seg_tree.dir/src/tree.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/seg_tree.dir/src/tree.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /Users/hans/Dropbox/Uni/MasterThesis/code/cpp/edge-based/src/tree.cpp -o CMakeFiles/seg_tree.dir/src/tree.cpp.s

CMakeFiles/seg_tree.dir/src/tree.cpp.o.requires:
.PHONY : CMakeFiles/seg_tree.dir/src/tree.cpp.o.requires

CMakeFiles/seg_tree.dir/src/tree.cpp.o.provides: CMakeFiles/seg_tree.dir/src/tree.cpp.o.requires
	$(MAKE) -f CMakeFiles/seg_tree.dir/build.make CMakeFiles/seg_tree.dir/src/tree.cpp.o.provides.build
.PHONY : CMakeFiles/seg_tree.dir/src/tree.cpp.o.provides

CMakeFiles/seg_tree.dir/src/tree.cpp.o.provides.build: CMakeFiles/seg_tree.dir/src/tree.cpp.o

# Object files for target seg_tree
seg_tree_OBJECTS = \
"CMakeFiles/seg_tree.dir/src/tree.cpp.o"

# External object files for target seg_tree
seg_tree_EXTERNAL_OBJECTS =

seg_tree: CMakeFiles/seg_tree.dir/src/tree.cpp.o
seg_tree: CMakeFiles/seg_tree.dir/build.make
seg_tree: /usr/local/lib/libboost_python-mt.a
seg_tree: /opt/ros/hydro/install_isolated/lib/libopencv_viz.3.0.0.dylib
seg_tree: /opt/ros/hydro/install_isolated/lib/libopencv_videostab.3.0.0.dylib
seg_tree: /opt/ros/hydro/install_isolated/lib/libopencv_video.3.0.0.dylib
seg_tree: /opt/ros/hydro/install_isolated/lib/libopencv_ts.a
seg_tree: /opt/ros/hydro/install_isolated/lib/libopencv_superres.3.0.0.dylib
seg_tree: /opt/ros/hydro/install_isolated/lib/libopencv_stitching.3.0.0.dylib
seg_tree: /opt/ros/hydro/install_isolated/lib/libopencv_softcascade.3.0.0.dylib
seg_tree: /opt/ros/hydro/install_isolated/lib/libopencv_shape.3.0.0.dylib
seg_tree: /opt/ros/hydro/install_isolated/lib/libopencv_photo.3.0.0.dylib
seg_tree: /opt/ros/hydro/install_isolated/lib/libopencv_optim.3.0.0.dylib
seg_tree: /opt/ros/hydro/install_isolated/lib/libopencv_objdetect.3.0.0.dylib
seg_tree: /opt/ros/hydro/install_isolated/lib/libopencv_nonfree.3.0.0.dylib
seg_tree: /opt/ros/hydro/install_isolated/lib/libopencv_ml.3.0.0.dylib
seg_tree: /opt/ros/hydro/install_isolated/lib/libopencv_legacy.3.0.0.dylib
seg_tree: /opt/ros/hydro/install_isolated/lib/libopencv_imgproc.3.0.0.dylib
seg_tree: /opt/ros/hydro/install_isolated/lib/libopencv_highgui.3.0.0.dylib
seg_tree: /opt/ros/hydro/install_isolated/lib/libopencv_flann.3.0.0.dylib
seg_tree: /opt/ros/hydro/install_isolated/lib/libopencv_features2d.3.0.0.dylib
seg_tree: /opt/ros/hydro/install_isolated/lib/libopencv_cudawarping.3.0.0.dylib
seg_tree: /opt/ros/hydro/install_isolated/lib/libopencv_cudastereo.3.0.0.dylib
seg_tree: /opt/ros/hydro/install_isolated/lib/libopencv_cudaoptflow.3.0.0.dylib
seg_tree: /opt/ros/hydro/install_isolated/lib/libopencv_cudaimgproc.3.0.0.dylib
seg_tree: /opt/ros/hydro/install_isolated/lib/libopencv_cudafilters.3.0.0.dylib
seg_tree: /opt/ros/hydro/install_isolated/lib/libopencv_cudafeatures2d.3.0.0.dylib
seg_tree: /opt/ros/hydro/install_isolated/lib/libopencv_cudabgsegm.3.0.0.dylib
seg_tree: /opt/ros/hydro/install_isolated/lib/libopencv_cudaarithm.3.0.0.dylib
seg_tree: /opt/ros/hydro/install_isolated/lib/libopencv_cuda.3.0.0.dylib
seg_tree: /opt/ros/hydro/install_isolated/lib/libopencv_core.3.0.0.dylib
seg_tree: /opt/ros/hydro/install_isolated/lib/libopencv_contrib.3.0.0.dylib
seg_tree: /opt/ros/hydro/install_isolated/lib/libopencv_calib3d.3.0.0.dylib
seg_tree: /usr/lib/libpython2.7.dylib
seg_tree: /opt/ros/hydro/install_isolated/lib/libopencv_cudawarping.3.0.0.dylib
seg_tree: /opt/ros/hydro/install_isolated/lib/libopencv_legacy.3.0.0.dylib
seg_tree: /opt/ros/hydro/install_isolated/lib/libopencv_cudaimgproc.3.0.0.dylib
seg_tree: /opt/ros/hydro/install_isolated/lib/libopencv_cudafilters.3.0.0.dylib
seg_tree: /opt/ros/hydro/install_isolated/lib/libopencv_video.3.0.0.dylib
seg_tree: /opt/ros/hydro/install_isolated/lib/libopencv_objdetect.3.0.0.dylib
seg_tree: /opt/ros/hydro/install_isolated/lib/libopencv_nonfree.3.0.0.dylib
seg_tree: /opt/ros/hydro/install_isolated/lib/libopencv_ml.3.0.0.dylib
seg_tree: /opt/ros/hydro/install_isolated/lib/libopencv_cudaarithm.3.0.0.dylib
seg_tree: /opt/ros/hydro/install_isolated/lib/libopencv_calib3d.3.0.0.dylib
seg_tree: /opt/ros/hydro/install_isolated/lib/libopencv_features2d.3.0.0.dylib
seg_tree: /opt/ros/hydro/install_isolated/lib/libopencv_highgui.3.0.0.dylib
seg_tree: /opt/ros/hydro/install_isolated/lib/libopencv_imgproc.3.0.0.dylib
seg_tree: /opt/ros/hydro/install_isolated/lib/libopencv_flann.3.0.0.dylib
seg_tree: /opt/ros/hydro/install_isolated/lib/libopencv_core.3.0.0.dylib
seg_tree: CMakeFiles/seg_tree.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable seg_tree"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/seg_tree.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/seg_tree.dir/build: seg_tree
.PHONY : CMakeFiles/seg_tree.dir/build

CMakeFiles/seg_tree.dir/requires: CMakeFiles/seg_tree.dir/src/tree.cpp.o.requires
.PHONY : CMakeFiles/seg_tree.dir/requires

CMakeFiles/seg_tree.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/seg_tree.dir/cmake_clean.cmake
.PHONY : CMakeFiles/seg_tree.dir/clean

CMakeFiles/seg_tree.dir/depend:
	cd /Users/hans/Dropbox/Uni/MasterThesis/code/cpp/edge-based/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/hans/Dropbox/Uni/MasterThesis/code/cpp/edge-based /Users/hans/Dropbox/Uni/MasterThesis/code/cpp/edge-based /Users/hans/Dropbox/Uni/MasterThesis/code/cpp/edge-based/build /Users/hans/Dropbox/Uni/MasterThesis/code/cpp/edge-based/build /Users/hans/Dropbox/Uni/MasterThesis/code/cpp/edge-based/build/CMakeFiles/seg_tree.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/seg_tree.dir/depend
