cmake_minimum_required(VERSION 3.0.2)

project(kcf_tracker)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Wextra -pedantic -Wno-long-long -fno-omit-frame-pointer")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(default_build_type "Release")
  message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE
      STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

FIND_PACKAGE( OpenCV REQUIRED )
link_directories ( ${OpenCV_LIB_DIR} )
include_directories ( ${OpenCV_INCLUDE_DIRS} )
MESSAGE(STATUS "OpenCV_LIB_DIR: ${OpenCV_LIB_DIR} ")
MESSAGE(STATUS "OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")

INCLUDE_DIRECTORIES( ${CMAKE_BINARY_DIR}/)
include_directories(${CMAKE_SOURCE_DIR}/src/ ${CMAKE_SOURCE_DIR}/src/piotr_fhog ${CMAKE_SOURCE_DIR}/src/cn)

include_directories(${CMAKE_SOURCE_DIR}/sse2neon)

add_subdirectory(src)

IF(NOT use_cuda)
  add_executable(kcf_vot main_vot.cpp vot.hpp videoio.cpp)
ELSE()
  cuda_add_executable( kcf_vot main_vot.cpp vot.hpp )
  target_link_libraries(kcf_vot ${CUDA_LIBRARIES})
ENDIF() #OPENCV_CUFFT

target_link_libraries(kcf_vot ${OpenCV_LIBS} kcf)
