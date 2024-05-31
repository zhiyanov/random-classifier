#pragma once

#include <unordered_set>
#include <vector>
#include <algorithm>
#include <numeric>
#include <iterator>

// declaration
template <class Iter>
class Combinations {
using T = typename std::iterator_traits<Iter>::value_type;

public:
    Combinations(Iter begin, Iter end, size_t size);
    
    bool Next();
    std::vector<T> Get() const;
    const std::vector<T>& Data() const;

private:
    std::vector<T> data_;
    size_t size_;

    std::vector<size_t> idx_;
};

template <class Iter>
class Permutations : public Combinations<Iter> {
using T = typename std::iterator_traits<Iter>::value_type;

public:
    Permutations(Iter begin, Iter end, size_t size);
    
    bool Next();
    std::vector<T> Get() const;

private:
    std::vector<T> perm_;
};

Combinations<std::vector<int>::iterator> Combs(size_t length, size_t size);
Permutations<std::vector<int>::iterator> Perms(size_t length, size_t size);

template <class IterA, class IterB>
std::vector<typename std::iterator_traits<IterA>::value_type> Residual(
    IterA from_begin, IterA from_end, IterB to_begin, IterB to_end);

size_t Binom(size_t n, size_t k);

// implementation
template <class Iter>
Combinations<Iter>::Combinations(Iter begin, Iter end, size_t size) : data_{begin, end}, size_{size} {
    idx_.assign(data_.size(), 0);
    std::fill(idx_.begin(), idx_.begin() + size, 1);
}

template <class Iter>
bool Combinations<Iter>::Next() {
    return std::prev_permutation(idx_.begin(), idx_.end());
}

template <class Iter>
std::vector<typename Combinations<Iter>::T> Combinations<Iter>::Get() const {
    std::vector<T> perm;
    for (size_t index = 0; index < idx_.size(); ++index) {
        if (idx_[index]) {
            perm.push_back(data_[index]);
        }
    }

    return perm;
}

template <class Iter>
const std::vector<typename Combinations<Iter>::T>& Combinations<Iter>::Data() const {
    return data_;
}

template <class Iter>
Permutations<Iter>::Permutations(Iter begin, Iter end, size_t size) : Combinations<Iter>(begin, end, size) {
    perm_ = Combinations<Iter>::Get();
}

template <class Iter>
bool Permutations<Iter>::Next() {
    if (std::next_permutation(perm_.begin(), perm_.end())) {
        return true;
    }

    auto next = Combinations<Iter>::Next();
    perm_ = Combinations<Iter>::Get();
    return next;
}

template <class Iter>
std::vector<typename Permutations<Iter>::T> Permutations<Iter>::Get() const {
    return perm_;
}

template <class IterA, class IterB>
std::vector<typename std::iterator_traits<IterA>::value_type> Residual(
    IterA from_begin, IterA from_end, IterB to_begin, IterB to_end) {
    std::unordered_set set(to_begin, to_end);
    
    std::vector<typename std::iterator_traits<IterA>::value_type> residual;
    for (auto iter = from_begin; iter != from_end; iter++) {
        if (set.find(*iter) == set.end()) {
            residual.push_back(*iter);
        }
    }

    return residual;
}