# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


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
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/kumarm/F2021/ml/hw12/solutions/q3_4

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/kumarm/F2021/ml/hw12/solutions/q3_4

# Include any dependencies generated for this target.
include CMakeFiles/q3_4.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/q3_4.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/q3_4.dir/flags.make

CMakeFiles/q3_4.dir/q3_4.cpp.o: CMakeFiles/q3_4.dir/flags.make
CMakeFiles/q3_4.dir/q3_4.cpp.o: q3_4.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/kumarm/F2021/ml/hw12/solutions/q3_4/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/q3_4.dir/q3_4.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/q3_4.dir/q3_4.cpp.o -c /home/kumarm/F2021/ml/hw12/solutions/q3_4/q3_4.cpp

CMakeFiles/q3_4.dir/q3_4.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/q3_4.dir/q3_4.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/kumarm/F2021/ml/hw12/solutions/q3_4/q3_4.cpp > CMakeFiles/q3_4.dir/q3_4.cpp.i

CMakeFiles/q3_4.dir/q3_4.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/q3_4.dir/q3_4.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/kumarm/F2021/ml/hw12/solutions/q3_4/q3_4.cpp -o CMakeFiles/q3_4.dir/q3_4.cpp.s

# Object files for target q3_4
q3_4_OBJECTS = \
"CMakeFiles/q3_4.dir/q3_4.cpp.o"

# External object files for target q3_4
q3_4_EXTERNAL_OBJECTS =

q3_4: CMakeFiles/q3_4.dir/q3_4.cpp.o
q3_4: CMakeFiles/q3_4.dir/build.make
q3_4: /usr/local/lib/libosqp.so
q3_4: CMakeFiles/q3_4.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/kumarm/F2021/ml/hw12/solutions/q3_4/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable q3_4"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/q3_4.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/q3_4.dir/build: q3_4

.PHONY : CMakeFiles/q3_4.dir/build

CMakeFiles/q3_4.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/q3_4.dir/cmake_clean.cmake
.PHONY : CMakeFiles/q3_4.dir/clean

CMakeFiles/q3_4.dir/depend:
	cd /home/kumarm/F2021/ml/hw12/solutions/q3_4 && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/kumarm/F2021/ml/hw12/solutions/q3_4 /home/kumarm/F2021/ml/hw12/solutions/q3_4 /home/kumarm/F2021/ml/hw12/solutions/q3_4 /home/kumarm/F2021/ml/hw12/solutions/q3_4 /home/kumarm/F2021/ml/hw12/solutions/q3_4/CMakeFiles/q3_4.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/q3_4.dir/depend

