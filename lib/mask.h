#pragma once

#include <cstddef>
#include <vector>
#include <array>

// mask
class ShortMask {
public:
    ShortMask(const std::vector<int>& indexes);

    bool operator<(const ShortMask& rhs) const;

private:
    static constexpr size_t kSize_ = sizeof(size_t) * 8;

    size_t mask_;
};
// mask

// long mask
template<size_t size = 1>
class LongMask {
public:
    LongMask(const std::vector<int>& indexes);

    bool operator<(const LongMask<size>& rhs) const;

private:
    static constexpr size_t kSize_ = sizeof(size_t) * 8;

    std::array<size_t, size> mask_;
};
// long mask

// implementation
template<size_t size>
LongMask<size>::LongMask(const std::vector<int>& indexes) {
    mask_.fill(0);

    for (auto index : indexes) {
        auto shift = index % kSize_;
        mask_[index / kSize_] += (static_cast<size_t>(1) << shift);
    }
}

template<size_t size>
bool LongMask<size>::operator<(const LongMask<size>& rhs) const {
    return mask_ < rhs.mask_;
}