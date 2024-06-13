#pragma once

#include <cstddef>
#include <vector>

// mask
class Mask {
public:
    Mask(const std::vector<int>& indexes);

    bool operator<(const Mask& rhs) const;

private:
    static constexpr size_t kSize_ = sizeof(size_t) * 8;

    size_t mask_;
};
// mask
