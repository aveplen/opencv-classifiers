# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.27

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /opt/homebrew/Cellar/cmake/3.27.1/bin/cmake

# The command to remove a file.
RM = /opt/homebrew/Cellar/cmake/3.27.1/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/plenkinav/Projects/opencv-classifiers

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/plenkinav/Projects/opencv-classifiers

# Include any dependencies generated for this target.
include CMakeFiles/opencv-classifiers.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/opencv-classifiers.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/opencv-classifiers.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/opencv-classifiers.dir/flags.make

CMakeFiles/opencv-classifiers.dir/main.cpp.o: CMakeFiles/opencv-classifiers.dir/flags.make
CMakeFiles/opencv-classifiers.dir/main.cpp.o: main.cpp
CMakeFiles/opencv-classifiers.dir/main.cpp.o: CMakeFiles/opencv-classifiers.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/plenkinav/Projects/opencv-classifiers/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/opencv-classifiers.dir/main.cpp.o"
	/Applications/Xcode-14.3.1.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/opencv-classifiers.dir/main.cpp.o -MF CMakeFiles/opencv-classifiers.dir/main.cpp.o.d -o CMakeFiles/opencv-classifiers.dir/main.cpp.o -c /Users/plenkinav/Projects/opencv-classifiers/main.cpp

CMakeFiles/opencv-classifiers.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/opencv-classifiers.dir/main.cpp.i"
	/Applications/Xcode-14.3.1.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/plenkinav/Projects/opencv-classifiers/main.cpp > CMakeFiles/opencv-classifiers.dir/main.cpp.i

CMakeFiles/opencv-classifiers.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/opencv-classifiers.dir/main.cpp.s"
	/Applications/Xcode-14.3.1.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/plenkinav/Projects/opencv-classifiers/main.cpp -o CMakeFiles/opencv-classifiers.dir/main.cpp.s

CMakeFiles/opencv-classifiers.dir/data.cpp.o: CMakeFiles/opencv-classifiers.dir/flags.make
CMakeFiles/opencv-classifiers.dir/data.cpp.o: data.cpp
CMakeFiles/opencv-classifiers.dir/data.cpp.o: CMakeFiles/opencv-classifiers.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/plenkinav/Projects/opencv-classifiers/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/opencv-classifiers.dir/data.cpp.o"
	/Applications/Xcode-14.3.1.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/opencv-classifiers.dir/data.cpp.o -MF CMakeFiles/opencv-classifiers.dir/data.cpp.o.d -o CMakeFiles/opencv-classifiers.dir/data.cpp.o -c /Users/plenkinav/Projects/opencv-classifiers/data.cpp

CMakeFiles/opencv-classifiers.dir/data.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/opencv-classifiers.dir/data.cpp.i"
	/Applications/Xcode-14.3.1.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/plenkinav/Projects/opencv-classifiers/data.cpp > CMakeFiles/opencv-classifiers.dir/data.cpp.i

CMakeFiles/opencv-classifiers.dir/data.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/opencv-classifiers.dir/data.cpp.s"
	/Applications/Xcode-14.3.1.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/plenkinav/Projects/opencv-classifiers/data.cpp -o CMakeFiles/opencv-classifiers.dir/data.cpp.s

CMakeFiles/opencv-classifiers.dir/statements_handler.cpp.o: CMakeFiles/opencv-classifiers.dir/flags.make
CMakeFiles/opencv-classifiers.dir/statements_handler.cpp.o: statements_handler.cpp
CMakeFiles/opencv-classifiers.dir/statements_handler.cpp.o: CMakeFiles/opencv-classifiers.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/plenkinav/Projects/opencv-classifiers/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/opencv-classifiers.dir/statements_handler.cpp.o"
	/Applications/Xcode-14.3.1.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/opencv-classifiers.dir/statements_handler.cpp.o -MF CMakeFiles/opencv-classifiers.dir/statements_handler.cpp.o.d -o CMakeFiles/opencv-classifiers.dir/statements_handler.cpp.o -c /Users/plenkinav/Projects/opencv-classifiers/statements_handler.cpp

CMakeFiles/opencv-classifiers.dir/statements_handler.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/opencv-classifiers.dir/statements_handler.cpp.i"
	/Applications/Xcode-14.3.1.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/plenkinav/Projects/opencv-classifiers/statements_handler.cpp > CMakeFiles/opencv-classifiers.dir/statements_handler.cpp.i

CMakeFiles/opencv-classifiers.dir/statements_handler.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/opencv-classifiers.dir/statements_handler.cpp.s"
	/Applications/Xcode-14.3.1.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/plenkinav/Projects/opencv-classifiers/statements_handler.cpp -o CMakeFiles/opencv-classifiers.dir/statements_handler.cpp.s

CMakeFiles/opencv-classifiers.dir/stacktrace.cpp.o: CMakeFiles/opencv-classifiers.dir/flags.make
CMakeFiles/opencv-classifiers.dir/stacktrace.cpp.o: stacktrace.cpp
CMakeFiles/opencv-classifiers.dir/stacktrace.cpp.o: CMakeFiles/opencv-classifiers.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/plenkinav/Projects/opencv-classifiers/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/opencv-classifiers.dir/stacktrace.cpp.o"
	/Applications/Xcode-14.3.1.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/opencv-classifiers.dir/stacktrace.cpp.o -MF CMakeFiles/opencv-classifiers.dir/stacktrace.cpp.o.d -o CMakeFiles/opencv-classifiers.dir/stacktrace.cpp.o -c /Users/plenkinav/Projects/opencv-classifiers/stacktrace.cpp

CMakeFiles/opencv-classifiers.dir/stacktrace.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/opencv-classifiers.dir/stacktrace.cpp.i"
	/Applications/Xcode-14.3.1.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/plenkinav/Projects/opencv-classifiers/stacktrace.cpp > CMakeFiles/opencv-classifiers.dir/stacktrace.cpp.i

CMakeFiles/opencv-classifiers.dir/stacktrace.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/opencv-classifiers.dir/stacktrace.cpp.s"
	/Applications/Xcode-14.3.1.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/plenkinav/Projects/opencv-classifiers/stacktrace.cpp -o CMakeFiles/opencv-classifiers.dir/stacktrace.cpp.s

# Object files for target opencv-classifiers
opencv__classifiers_OBJECTS = \
"CMakeFiles/opencv-classifiers.dir/main.cpp.o" \
"CMakeFiles/opencv-classifiers.dir/data.cpp.o" \
"CMakeFiles/opencv-classifiers.dir/statements_handler.cpp.o" \
"CMakeFiles/opencv-classifiers.dir/stacktrace.cpp.o"

# External object files for target opencv-classifiers
opencv__classifiers_EXTERNAL_OBJECTS =

opencv-classifiers: CMakeFiles/opencv-classifiers.dir/main.cpp.o
opencv-classifiers: CMakeFiles/opencv-classifiers.dir/data.cpp.o
opencv-classifiers: CMakeFiles/opencv-classifiers.dir/statements_handler.cpp.o
opencv-classifiers: CMakeFiles/opencv-classifiers.dir/stacktrace.cpp.o
opencv-classifiers: CMakeFiles/opencv-classifiers.dir/build.make
opencv-classifiers: /opt/homebrew/lib/libopencv_gapi.4.9.0.dylib
opencv-classifiers: /opt/homebrew/lib/libopencv_stitching.4.9.0.dylib
opencv-classifiers: /opt/homebrew/lib/libopencv_alphamat.4.9.0.dylib
opencv-classifiers: /opt/homebrew/lib/libopencv_aruco.4.9.0.dylib
opencv-classifiers: /opt/homebrew/lib/libopencv_bgsegm.4.9.0.dylib
opencv-classifiers: /opt/homebrew/lib/libopencv_bioinspired.4.9.0.dylib
opencv-classifiers: /opt/homebrew/lib/libopencv_ccalib.4.9.0.dylib
opencv-classifiers: /opt/homebrew/lib/libopencv_dnn_objdetect.4.9.0.dylib
opencv-classifiers: /opt/homebrew/lib/libopencv_dnn_superres.4.9.0.dylib
opencv-classifiers: /opt/homebrew/lib/libopencv_dpm.4.9.0.dylib
opencv-classifiers: /opt/homebrew/lib/libopencv_face.4.9.0.dylib
opencv-classifiers: /opt/homebrew/lib/libopencv_freetype.4.9.0.dylib
opencv-classifiers: /opt/homebrew/lib/libopencv_fuzzy.4.9.0.dylib
opencv-classifiers: /opt/homebrew/lib/libopencv_hfs.4.9.0.dylib
opencv-classifiers: /opt/homebrew/lib/libopencv_img_hash.4.9.0.dylib
opencv-classifiers: /opt/homebrew/lib/libopencv_intensity_transform.4.9.0.dylib
opencv-classifiers: /opt/homebrew/lib/libopencv_line_descriptor.4.9.0.dylib
opencv-classifiers: /opt/homebrew/lib/libopencv_mcc.4.9.0.dylib
opencv-classifiers: /opt/homebrew/lib/libopencv_quality.4.9.0.dylib
opencv-classifiers: /opt/homebrew/lib/libopencv_rapid.4.9.0.dylib
opencv-classifiers: /opt/homebrew/lib/libopencv_reg.4.9.0.dylib
opencv-classifiers: /opt/homebrew/lib/libopencv_rgbd.4.9.0.dylib
opencv-classifiers: /opt/homebrew/lib/libopencv_saliency.4.9.0.dylib
opencv-classifiers: /opt/homebrew/lib/libopencv_sfm.4.9.0.dylib
opencv-classifiers: /opt/homebrew/lib/libopencv_stereo.4.9.0.dylib
opencv-classifiers: /opt/homebrew/lib/libopencv_structured_light.4.9.0.dylib
opencv-classifiers: /opt/homebrew/lib/libopencv_superres.4.9.0.dylib
opencv-classifiers: /opt/homebrew/lib/libopencv_surface_matching.4.9.0.dylib
opencv-classifiers: /opt/homebrew/lib/libopencv_tracking.4.9.0.dylib
opencv-classifiers: /opt/homebrew/lib/libopencv_videostab.4.9.0.dylib
opencv-classifiers: /opt/homebrew/lib/libopencv_viz.4.9.0.dylib
opencv-classifiers: /opt/homebrew/lib/libopencv_wechat_qrcode.4.9.0.dylib
opencv-classifiers: /opt/homebrew/lib/libopencv_xfeatures2d.4.9.0.dylib
opencv-classifiers: /opt/homebrew/lib/libopencv_xobjdetect.4.9.0.dylib
opencv-classifiers: /opt/homebrew/lib/libopencv_xphoto.4.9.0.dylib
opencv-classifiers: _deps/yaml-cpp-build/libyaml-cpp.a
opencv-classifiers: _deps/logurugitrepo-build/libloguru.2.1.0.dylib
opencv-classifiers: boost-cmake/libboost_system.a
opencv-classifiers: boost-cmake/libboost_thread.a
opencv-classifiers: boost-cmake/libboost_log.a
opencv-classifiers: boost-cmake/libboost_program_options.a
opencv-classifiers: boost-cmake/libboost_chrono.a
opencv-classifiers: boost-cmake/libboost_exception.a
opencv-classifiers: lib/libPocoUtil.101.dylib
opencv-classifiers: lib/libPocoDataSQLite.101.dylib
opencv-classifiers: lib/libPocoEncodings.101.dylib
opencv-classifiers: lib/libPocoJSON.101.dylib
opencv-classifiers: lib/libPocoMongoDB.101.dylib
opencv-classifiers: lib/libPocoRedis.101.dylib
opencv-classifiers: lib/libPocoXML.101.dylib
opencv-classifiers: lib/libPocoZip.101.dylib
opencv-classifiers: /opt/homebrew/lib/libopencv_shape.4.9.0.dylib
opencv-classifiers: /opt/homebrew/lib/libopencv_highgui.4.9.0.dylib
opencv-classifiers: /opt/homebrew/lib/libopencv_datasets.4.9.0.dylib
opencv-classifiers: /opt/homebrew/lib/libopencv_plot.4.9.0.dylib
opencv-classifiers: /opt/homebrew/lib/libopencv_text.4.9.0.dylib
opencv-classifiers: /opt/homebrew/lib/libopencv_ml.4.9.0.dylib
opencv-classifiers: /opt/homebrew/lib/libopencv_phase_unwrapping.4.9.0.dylib
opencv-classifiers: /opt/homebrew/lib/libopencv_optflow.4.9.0.dylib
opencv-classifiers: /opt/homebrew/lib/libopencv_ximgproc.4.9.0.dylib
opencv-classifiers: /opt/homebrew/lib/libopencv_video.4.9.0.dylib
opencv-classifiers: /opt/homebrew/lib/libopencv_videoio.4.9.0.dylib
opencv-classifiers: /opt/homebrew/lib/libopencv_imgcodecs.4.9.0.dylib
opencv-classifiers: /opt/homebrew/lib/libopencv_objdetect.4.9.0.dylib
opencv-classifiers: /opt/homebrew/lib/libopencv_calib3d.4.9.0.dylib
opencv-classifiers: /opt/homebrew/lib/libopencv_dnn.4.9.0.dylib
opencv-classifiers: /opt/homebrew/lib/libopencv_features2d.4.9.0.dylib
opencv-classifiers: /opt/homebrew/lib/libopencv_flann.4.9.0.dylib
opencv-classifiers: /opt/homebrew/lib/libopencv_photo.4.9.0.dylib
opencv-classifiers: /opt/homebrew/lib/libopencv_imgproc.4.9.0.dylib
opencv-classifiers: /opt/homebrew/lib/libopencv_core.4.9.0.dylib
opencv-classifiers: boost-cmake/libboost_thread.a
opencv-classifiers: boost-cmake/libboost_chrono.a
opencv-classifiers: boost-cmake/libboost_atomic.a
opencv-classifiers: boost-cmake/libboost_date_time.a
opencv-classifiers: boost-cmake/libboost_filesystem.a
opencv-classifiers: lib/libPocoData.101.dylib
opencv-classifiers: lib/libPocoNet.101.dylib
opencv-classifiers: lib/libPocoFoundation.101.dylib
opencv-classifiers: CMakeFiles/opencv-classifiers.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/Users/plenkinav/Projects/opencv-classifiers/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Linking CXX executable opencv-classifiers"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/opencv-classifiers.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/opencv-classifiers.dir/build: opencv-classifiers
.PHONY : CMakeFiles/opencv-classifiers.dir/build

CMakeFiles/opencv-classifiers.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/opencv-classifiers.dir/cmake_clean.cmake
.PHONY : CMakeFiles/opencv-classifiers.dir/clean

CMakeFiles/opencv-classifiers.dir/depend:
	cd /Users/plenkinav/Projects/opencv-classifiers && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/plenkinav/Projects/opencv-classifiers /Users/plenkinav/Projects/opencv-classifiers /Users/plenkinav/Projects/opencv-classifiers /Users/plenkinav/Projects/opencv-classifiers /Users/plenkinav/Projects/opencv-classifiers/CMakeFiles/opencv-classifiers.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/opencv-classifiers.dir/depend

