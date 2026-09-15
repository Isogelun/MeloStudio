#pragma once

#include <compare>
#include <cstdint>

namespace melo::core {

struct ProjectTick final {
    std::int64_t value{};

    friend constexpr auto operator<=>(const ProjectTick&, const ProjectTick&) = default;
};

struct SampleFrame final {
    std::int64_t value{};

    friend constexpr auto operator<=>(const SampleFrame&, const SampleFrame&) = default;
};

struct Seconds final {
    double value{};
};

struct TickRange final {
    ProjectTick start;
    ProjectTick end;

    [[nodiscard]] constexpr bool valid() const noexcept {
        return start < end;
    }

    [[nodiscard]] constexpr bool contains(ProjectTick tick) const noexcept {
        return start <= tick && tick < end;
    }
};

} // namespace melo::core
