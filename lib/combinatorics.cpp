#include "combinatorics.h"

Combinations<std::vector<int>::iterator> Combs(size_t length, size_t size) {
    std::vector<int> arr;
    arr.assign(length, 0);
    std::iota(arr.begin(), arr.end(), 0);

    return Combinations{arr.begin(), arr.end(), size};
}

Permutations<std::vector<int>::iterator> Perms(size_t length, size_t size) {
    std::vector<int> arr;
    arr.assign(length, 0);
    std::iota(arr.begin(), arr.end(), 0);

    return Permutations{arr.begin(), arr.end(), size};
}

size_t Binom(size_t n, size_t k) {
    if (n < k) {
        return 0;
    } else if (!n || !k) {
        return 1;
    }

    return Binom(n - 1, k - 1) + Binom(n - 1, k);
}