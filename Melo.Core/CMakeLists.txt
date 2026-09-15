cmake_minimum_required(VERSION 3.28)

project(
    MeloCore
    VERSION 0.1.0
    DESCRIPTION "Portable core library for MeloStudio"
    LANGUAGES CXX
)

include(CTest)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

add_library(melo_core STATIC)
add_library(Melo::Core ALIAS melo_core)

target_sources(
    melo_core
    PRIVATE
        src/version.cpp
    PUBLIC
        FILE_SET public_headers
        TYPE HEADERS
        BASE_DIRS include
        FILES
            include/melo/core/core.hpp
            include/melo/core/id.hpp
            include/melo/core/time.hpp
            include/melo/core/version.hpp
)

target_compile_features(melo_core PUBLIC cxx_std_20)
set_target_properties(
    melo_core
    PROPERTIES
        CXX_EXTENSIONS OFF
        OUTPUT_NAME "Melo.Core"
        EXPORT_NAME Core
        VERSION "${PROJECT_VERSION}"
)

if(MSVC)
    target_compile_options(melo_core PRIVATE /W4 /permissive- /Zc:__cplusplus /utf-8)
else()
    target_compile_options(melo_core PRIVATE -Wall -Wextra -Wpedantic -Wconversion)
endif()

if(BUILD_TESTING)
    add_subdirectory(tests)
endif()

configure_package_config_file(
    cmake/MeloCoreConfig.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/MeloCoreConfig.cmake"
    INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/MeloCore"
)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/MeloCoreConfigVersion.cmake"
    VERSION "${PROJECT_VERSION}"
    COMPATIBILITY SameMajorVersion
)

install(
    TARGETS melo_core
    EXPORT MeloCoreTargets
    FILE_SET public_headers DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
)

install(
    EXPORT MeloCoreTargets
    FILE MeloCoreTargets.cmake
    NAMESPACE Melo::
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/MeloCore"
)

install(
    FILES
        "${CMAKE_CURRENT_BINARY_DIR}/MeloCoreConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/MeloCoreConfigVersion.cmake"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/MeloCore"
)
