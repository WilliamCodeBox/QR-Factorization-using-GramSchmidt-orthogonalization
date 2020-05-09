#################################################################
###                         Options                           ###
#################################################################
option(BUILD_TESTS "Build the tests" OFF)
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

#################################################################
###                         Project                           ###
#################################################################
cmake_minimum_required(VERSION 3.7)
include               (cmake/HunterGate.cmake)
HunterGate            ( URL "https://github.com/cpp-pm/hunter/archive/v0.23.253.tar.gz"
                        SHA1 "88ea6d37c897a81a080eb9ae0f69d7807bbb3c73")
project               (YourGreatProjectName VERSION 1.0.0 LANGUAGES CXX)

#################################################################
###                         Guards                            ###
#################################################################
if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
    message(WARNING "In-source build detected")
    message(STATUS "Out-of-source build is preferred")
    message(STATUS "   Create a separate directory, named `build` for convention")
    message(STATUS "   and run CMake again")
endif()

#################################################################
###                       Dependencies                        ###
#################################################################
find_package(Armadillo CONFIG REQUIRED)



#################################################################
###                         Sources                           ###
#################################################################
set(
    QRFAC_SOURCE_FILES
    ${PROJECT_SOURCE_DIR}/source/gram-schmidt.cpp)


#################################################################
###                         Targets                           ###
#################################################################
set                       (PROJECT_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)
add_library               (qrfac SHARED ${QRFAC_SOURCE_FILES})
target_include_directories(qrfac PUBLIC ${PROJECT_INCLUDE_DIR})
target_link_libraries     (qrfac PRIVATE ${ARMADILLO_LIBRARIES})
set_target_properties     (qrfac PROPERTIES CXX_STANDARD 11)


#################################################################
###                        Install                            ###
#################################################################



#################################################################
###                          Tests                            ###
#################################################################
if(BUILD_TESTS)
    add_subdirectory(test)
endif()

