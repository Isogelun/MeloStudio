add_executable(melo_core_tests core_tests.cpp)
target_link_libraries(melo_core_tests PRIVATE Melo::Core)
target_compile_features(melo_core_tests PRIVATE cxx_std_20)

if(MSVC)
    target_compile_options(melo_core_tests PRIVATE /W4 /permissive- /Zc:__cplusplus /utf-8)
else()
    target_compile_options(melo_core_tests PRIVATE -Wall -Wextra -Wpedantic -Wconversion)
endif()

add_test(NAME MeloCore.UnitTests COMMAND melo_core_tests)
