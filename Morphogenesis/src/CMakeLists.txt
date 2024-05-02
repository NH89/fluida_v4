cmake_minimum_required(VERSION 3.10)
set(GCC_COVERAGE_COMPILE_FLAGS "-m32")
cmake_policy(SET CMP0054 NEW)

project(Morphogenesis LANGUAGES C CXX)

#set(PROJECT_NAME createDemo2)
set(CMAKE_C_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING
      "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel."
      FORCE)
endif(NOT CMAKE_BUILD_TYPE)

set(Boost_USE_STATIC_LIBS OFF)
#set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)

find_package(Boost  REQUIRED 
    COMPONENTS  filesystem
                #thread
    ) # see https://cliutils.gitlab.io/modern-cmake/chapters/packages/Boost.html
message(STATUS "Boost version: ${Boost_VERSION}")

# This is needed if your Boost version is newer than your CMake version
# or if you have an old version of CMake (<3.5)
if(NOT TARGET Boost::filesystem)
    add_library(Boost::filesystem IMPORTED INTERFACE)
    set_property(TARGET Boost::filesystem PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${Boost_INCLUDE_DIR})
    set_property(TARGET Boost::filesystem PROPERTY INTERFACE_LINK_LIBRARIES      ${Boost_LIBRARIES})
endif()

find_package(OpenCL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(jsoncpp REQUIRED)
find_package(PkgConfig REQUIRED)
#find_package(Threads REQUIRED)
find_package(VTK REQUIRED 
COMPONENTS 
    CommonCore
    IOGeometry
    IOLegacy
    IOXML
    InteractionStyle
)

if (NOT VTK_FOUND)
  message("VTK not found: ${VTK_NOT_FOUND_MESSAGE}")
  return ()
endif()
message (STATUS "VTK_VERSION: ${VTK_VERSION}")

if( OpenCV_FOUND )
  # Additional Include Directories
  include_directories( ${OpenCV_INCLUDE_DIRS} )

  # Additional Library Directories
  link_directories( ${OpenCV_LIB_DIR} )

  # Additional Dependencies
  
endif()

####

# set(CMAKE_CXX_COMPILER "/usr/bin/gcc")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I /usr/lib/gcc/x86_64-linux-gnu/11/include -g")
# set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGS}")

# set(VTK_DIR "/usr/local/include/vtk-9.3/")

###

set(CMAKE_THREAD_LIBS_INIT "-lpthread")
set(CMAKE_HAVE_THREADS_LIBRARY 1)
set(CMAKE_USE_PTHREADS_INIT 1)
set(THREADS_PREFER_PTHREAD_FLAG ON)



pkg_check_modules(JSONCPP jsoncpp)

include_directories(${JSONCPP_INCLUDE_DIRS})
include_directories(${VTK_INCLUDE_DIRS})



add_executable(${PROJECT_NAME}
        createDemo2.cpp
        )



target_link_libraries(${PROJECT_NAME} PRIVATE
        OpenCL::OpenCL
        ${JSONCPP_LIBRARIES}
        Boost::filesystem
        ${VTK_LIBRARIES}
        stdc++
        m #needed to be added after implementing UpdatePrams() ??
)

set_target_properties(${PROJECT_NAME} PROPERTIES C_STANDARD 17
                                                 C_STANDARD_REQUIRED ON
                                                 C_EXTENSIONS OFF)

target_compile_definitions(${PROJECT_NAME} PRIVATE CL_TARGET_OPENCL_VERSION=300)
