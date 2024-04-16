PROJECT(exercises)

CMAKE_MINIMUM_REQUIRED(VERSION 3.22)

set (CMAKE_CXX_STANDARD 17)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

list(APPEND CMAKE_PREFIX_PATH "/home/vvirkkal/Libraries/opencv")

find_package(OpenCV REQUIRED )
find_package(OpenMP REQUIRED )


add_subdirectory(prereq)
add_subdirectory(cp1)
add_subdirectory(is4)
add_subdirectory(mf1)