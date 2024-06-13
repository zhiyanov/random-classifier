#include "mask.h"

#include <vector>

Mask::Mask(const std::vector<int>& indexes) : mask_{0} {
    for (auto index : indexes) {
        auto shift = index % kSize_;
        mask_ += (static_cast<size_t>(1) << shift);
    }
}

bool Mask::operator<(const Mask& rhs) const {
    return mask_ < rhs.mask_;
}
// mask
