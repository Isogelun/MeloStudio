#pragma once

#include <compare>
#include <concepts>
#include <cstdint>

namespace melo::core {

template <typename Tag>
class Id final {
public:
    using value_type = std::uint64_t;

    constexpr Id() noexcept = default;
    explicit constexpr Id(value_type value) noexcept : value_(value) {}

    [[nodiscard]] constexpr value_type value() const noexcept { return value_; }
    [[nodiscard]] constexpr bool valid() const noexcept { return value_ != 0; }

    friend constexpr auto operator<=>(const Id&, const Id&) = default;

private:
    value_type value_{};
};

struct ProjectIdTag;
struct TrackIdTag;
struct PartIdTag;
struct NoteIdTag;

using ProjectId = Id<ProjectIdTag>;
using TrackId = Id<TrackIdTag>;
using PartId = Id<PartIdTag>;
using NoteId = Id<NoteIdTag>;

static_assert(std::regular<ProjectId>);

} // namespace melo::core
