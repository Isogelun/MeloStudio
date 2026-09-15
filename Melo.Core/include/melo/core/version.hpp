#pragma once

#include <string_view>

namespace melo::core {

struct Version final {
    int major;
    int minor;
    int patch;

    friend constexpr bool operator==(const Version&, const Version&) = default;
};

inline constexpr Version library_version{0, 1, 0};

[[nodiscard]] std::string_view version_string() noexcept;

} // namespace melo::core
