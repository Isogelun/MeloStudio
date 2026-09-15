#include <melo/core/core.hpp>

#include <cstdlib>
#include <iostream>
#include <string_view>

namespace {

int failures = 0;

void expect(bool condition, std::string_view message) {
    if (!condition) {
        std::cerr << "FAILED: " << message << '\n';
        ++failures;
    }
}

} // namespace

int main() {
    using namespace melo::core;

    expect(library_version == Version{0, 1, 0}, "library version components");
    expect(version_string() == "0.1.0", "library version string");

    expect(!NoteId{}.valid(), "default ID is invalid");
    expect(NoteId{42}.valid(), "non-zero ID is valid");
    expect(NoteId{42} == NoteId{42}, "IDs of the same type compare");

    const TickRange range{ProjectTick{10}, ProjectTick{20}};
    expect(range.valid(), "ascending tick range is valid");
    expect(range.contains(ProjectTick{10}), "range includes its start");
    expect(range.contains(ProjectTick{19}), "range includes interior ticks");
    expect(!range.contains(ProjectTick{20}), "range excludes its end");

    if (failures == 0) {
        std::cout << "All Melo.Core tests passed.\n";
        return EXIT_SUCCESS;
    }

    std::cerr << failures << " test(s) failed.\n";
    return EXIT_FAILURE;
}
