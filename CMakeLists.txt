cmake_minimum_required(VERSION 3.15...3.26)
project(vectextr LANGUAGES C CXX)

# Find required packages
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module NumPy)
find_package(pybind11 CONFIG REQUIRED)

# Create the extension module
pybind11_add_module(vectextr 
    extract_wrapper.cpp 
    extract.c
)

# Include directories
target_include_directories(vectextr PRIVATE 
    ${Python_NumPy_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Set C++ standard
target_compile_features(vectextr PRIVATE cxx_std_11)

# Set appropriate compiler flags
if(MSVC)
    target_compile_options(vectextr PRIVATE /W4)
else()
    target_compile_options(vectextr PRIVATE -Wall -Wextra)
endif()

# Install the extension
install(TARGETS vectextr DESTINATION ${SKBUILD_PLATLIB_DIR}/vectextr)

# Create __init__.py to make it a proper package
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/__init__.py" "from .vectextr import *\n")
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/__init__.py" DESTINATION ${SKBUILD_PLATLIB_DIR}/vectextr)
