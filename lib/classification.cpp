#include "classification.h"
#include "combinatorics.h"
#include "mask.h"

#include "eigen/Eigen/Dense"
#include "tqdm/tqdm.hpp"

#include <iostream>

#include <iterator>
#include <functional>
#include <optional>
#include <random>

#include <tuple>
#include <array>
#include <vector>
#include <set>

#include <cmath>
#include <algorithm>
#include <numeric>

#include <thread>
#include <mutex>

using Mask = LongMask<4>;

namespace eg = Eigen;

constexpr std::array kSeeds = {961, 221, 987, 109, 644, 181, 763, 59,  263, 922, 165, 531, 634,
                               350, 285, 158, 968, 807, 716, 348, 675, 679, 468, 396, 424, 286,
                               919, 253, 935, 752, 237, 73,  732, 9,   477, 39,  446, 55,  386,
                               326, 797, 22,  295, 362, 939, 319, 403, 789, 702, 964, 346, 887,
                               743, 235, 276, 631, 597, 772, 459, 738, 376, 146, 949, 901};

constexpr float kEpsilon = 1e-5;

constexpr size_t kBinom = 0;

template <class Iter, class Stream>
void Print(Iter begin, Iter end, Stream *stream) {
    for (auto iter = begin; iter != end; ++iter) {
        *stream << *iter << " ";
    }

    *stream << "\n";
}

// class
std::vector<Class> Reverse(const std::vector<Class> &y) {
    std::vector<Class> reverse{y};

    for (auto &c : reverse) {
        if (c == Class::Positive) {
            c = Class::Negative;
        } else if (c == Class::Negative) {
            c = Class::Positive;
        }
    }

    return reverse;
}
// class

// metrics
Confusion::Confusion(const std::vector<Class> &ground, const std::vector<Class> &pred) {
    for (size_t index = 0; index < ground.size(); ++index) {
        if (pred[index] == Class::Positive && ground[index] == Class::Positive) {
            tp_++;
        } else if (pred[index] == Class::Positive && ground[index] == Class::Negative) {
            fp_++;
        } else if (pred[index] == Class::Negative && ground[index] == Class::Positive) {
            fn_++;
        } else if (pred[index] == Class::Negative && ground[index] == Class::Negative) {
            tn_++;
        } else {
            un_++;
        }
    }
}

int Confusion::Error() const {
    return fp_ + fn_;
}

int Confusion::Accur() const {
    return tp_ + tn_ + un_;
}
// metrics

// normalization
eg::VectorXf Normalize(const eg::MatrixXf &X) {
    size_t dim = X.cols();

    const auto &point = X(0, eg::indexing::all);
    auto matrix = X.rowwise() - point;

    eg::BDCSVD<eg::MatrixXf, eg::ComputeFullV> svd{matrix};

    eg::VectorXf vec = eg::VectorXf::Zero(dim);
    vec(dim - 1) = 1.;

    return svd.matrixV() * vec;
}
// normalization

// linclass
LinearClassifier::LinearClassifier(const eg::VectorXf &point, const eg::VectorXf &normal)
    : point_{point}, normal_{normal} {
}

LinearClassifier::LinearClassifier(const eg::MatrixXf &X) {
    point_ = X(0, eg::indexing::all);
    normal_ = Normalize(X);
}

LinearClassifier LinearClassifier::Fit(const eg::MatrixXf &X,
                                       std::function<int(const std::vector<Class> &)> Loss) {
    size_t length = X.rows();
    size_t dim = X.cols();

    auto combs = Combs(length, dim);

    LinearClassifier best_clf{X(combs.Get(), eg::indexing::all)};
    auto best_loss = Loss(best_clf.Predict(X));

    while (combs.Next()) {
        LinearClassifier clf{X(combs.Get(), eg::indexing::all)};
        auto pred = clf.Predict(X);

        for (auto reverse : {false, true}) {
            if (reverse) {
                pred = Reverse(pred);
                clf = clf.Opposite();
            }

            auto loss = Loss(pred);
            if (loss < best_loss) {
                best_clf = clf;
                best_loss = loss;
            }
        }
    }

    return best_clf;
}

LinearClassifier LinearClassifier::Fit(const eg::MatrixXf &X,
                                       std::function<int(const std::vector<Class> &)> Loss,
                                       const std::vector<LinearClassifier> &clfs) {
    LinearClassifier best_clf = clfs.front();
    auto best_loss = Loss(best_clf.Predict(X));

    for (auto clf : clfs) {
        auto pred = clf.Predict(X);

        for (auto reverse : {false, true}) {
            if (reverse) {
                pred = Reverse(pred);
                clf = clf.Opposite();
            }

            auto loss = Loss(pred);
            if (loss < best_loss) {
                best_clf = clf;
                best_loss = loss;
            }
        }
    }

    return best_clf;
}

std::vector<Class> LinearClassifier::Predict(const eg::MatrixXf &X) const {
    int length = X.rows();

    std::vector<Class> pred;
    pred.assign(length, Class::Zero);

    auto product = (X.rowwise() - point_.transpose()) * normal_;

    for (int index = 0; index < length; ++index) {
        if (product(index) > kEpsilon) {
            pred[index] = Class::Positive;
        } else if (product(index) < -kEpsilon) {
            pred[index] = Class::Negative;
        } else {
            pred[index] = Class::Zero;
        }
    }

    return pred;
}

LinearClassifier LinearClassifier::Opposite() const {
    return {point_, -normal_};
}
// linclass

template <bool visualize>
std::tuple<size_t, size_t> Approximate(const eg::MatrixXf &X, std::vector<Class> y, size_t k,
                                       float eps, size_t seed) {
    size_t length = X.rows();
    size_t dim = X.cols();

    // build classifiers
    std::vector<LinearClassifier> clfs;
    auto combs = Combs(length, dim);
    do {
        clfs.emplace_back(X(combs.Get(), eg::indexing::all));
    } while (combs.Next());

    // iteration
    std::mt19937 rng(seed);

    size_t count = 0;
    size_t iters = static_cast<size_t>(1 / (eps * eps));

    auto step = [&] {
        std::shuffle(y.begin(), y.end(), rng);

        auto loss = [&y](const std::vector<Class> &pred) {
            auto conf = Confusion{y, pred};
            return conf.Error();
        };

        for (const auto &clf : clfs) {
            auto prediction = clf.Predict(X);

            if (loss(prediction) <= k || loss(Reverse(prediction)) <= k) {
                count++;
                break;
            }
        }
    };

    if constexpr (visualize) {
        for (auto iter : tq::trange(++iters)) {
            step();
        }

        std::cerr << "\n";
    } else {
        for (size_t iter = 0; iter < iters; ++iter) {
            step();
        }
    }

    return {count, iters};
}

std::optional<std::tuple<size_t, size_t, size_t, size_t>> Distribute(size_t p, size_t n, size_t t,
                                                                     size_t f, size_t k) {
    // true positive
    if (p * 2 + n < f + k || (p * 2 + n - f - k) % 2) {
        return std::nullopt;
    }
    auto tp = (p * 2 + n - f - k) / 2;

    // false positive
    if (f + k < n || (f + k - n) % 2) {
        return std::nullopt;
    }
    auto fp = (f + k - n) / 2;

    // false negative
    if (n + k < f || (n + k - f) % 2) {
        return std::nullopt;
    }
    auto fn = (n + k - f) / 2;

    // true negative
    if (f + n < k || (f + n - k) % 2) {
        return std::nullopt;
    }
    auto tn = (f + n - k) / 2;

    return {{tp, fp, fn, tn}};
}

template <bool visualize>
std::set<Mask> Exact(const eg::MatrixXf &X, const std::vector<Class> &y, size_t k,
                     const std::vector<LinearClassifier> &clfs) {
    size_t length = X.rows();
    size_t dim = X.cols();

    size_t truenum = 0, falsenum = 0;
    for (auto a : y) {
        if (a == Class::Positive) {
            truenum++;
        } else if (a == Class::Negative) {
            falsenum++;
        }
    }

    // iteration
    std::set<Mask> colors;

    auto step = [&](LinearClassifier clf) {
        auto pred = clf.Predict(X);

        for (auto reverse : {false, true}) {
            if (reverse) {
                pred = Reverse(pred);
                clf = clf.Opposite();
            }

            // predict
            std::vector<int> posinds, neginds, zeroinds;
            for (int i = 0; i < static_cast<int>(length); ++i) {
                if (pred[i] == Class::Positive) {
                    posinds.push_back(i);
                } else if (pred[i] == Class::Negative) {
                    neginds.push_back(i);
                } else {
                    zeroinds.push_back(i);
                }
            }

            // distribute zeros
            for (size_t zero2pos = 0; zero2pos <= zeroinds.size(); ++zero2pos) {
                auto zero2neg = zeroinds.size() - zero2pos;

                // distribute ground
                auto distribute = Distribute(posinds.size() + zero2pos, neginds.size() + zero2neg,
                                             truenum, falsenum, k);

                if (!distribute) {
                    continue;
                }

                auto [tp, fp, fn, tn] = distribute.value();

                auto zero_combs = Combinations(zeroinds.begin(), zeroinds.end(), zero2pos);
                do {
                    auto z2p_cmb = zero_combs.Get();
                    auto z2n_cmb =
                        Residual(zeroinds.begin(), zeroinds.end(), z2p_cmb.begin(), z2p_cmb.end());

                    auto positives = posinds;
                    positives.insert(positives.end(), z2p_cmb.begin(), z2p_cmb.end());
                    auto negatives = neginds;
                    negatives.insert(negatives.end(), z2n_cmb.begin(), z2n_cmb.end());

                    auto tp_combs = Combinations(positives.begin(), positives.end(), tp);
                    auto fn_combs = Combinations(negatives.begin(), negatives.end(), fn);
                    do {
                        auto tp_comb = tp_combs.Get();
                        do {
                            auto fn_comb = fn_combs.Get();

                            auto trues = tp_comb;
                            trues.insert(trues.end(), fn_comb.begin(), fn_comb.end());

                            // std::sort(trues.begin(), trues.end());
                            colors.emplace(trues);
                        } while (fn_combs.Next());
                    } while (tp_combs.Next());
                } while (zero_combs.Next());
            }
        }
    };

    if constexpr (visualize) {
        for (const auto &clf : tq::tqdm(clfs)) {
            step(clf);
        }

        std::cerr << "\n";
    } else {
        for (const auto &clf : clfs) {
            step(clf);
        }
    }

    return colors;
}

std::tuple<size_t, size_t> Approximate(const eg::MatrixXf &X, std::vector<Class> y, size_t k,
                                       float eps, size_t parallel) {
    std::vector<std::tuple<size_t, size_t>> drains(parallel);

    std::mutex mutex;
    std::vector<std::thread> threads;
    for (size_t index = 0; index < parallel; ++index) {
        threads.emplace_back([index, parallel, &mutex, &drains, &X, &y, k, eps]() {
            std::tuple<size_t, size_t> proba;

            if (index == parallel - 1) {
                proba = Approximate<true>(X, y, k, eps * std::sqrt(static_cast<float>(parallel)),
                                          kSeeds[index]);
            } else {
                proba = Approximate<false>(X, y, k, eps * std::sqrt(static_cast<float>(parallel)),
                                           kSeeds[index]);
            }

            std::lock_guard guard{mutex};
            std::get<0>(drains[index]) += std::get<0>(proba);
            std::get<1>(drains[index]) += std::get<1>(proba);
        });
    }

    for (auto &&thread : threads) {
        thread.join();
    }
    
    std::tuple<size_t, size_t> drain = {0, 0};
    for (auto index : tq::trange(parallel)) {
        std::get<0>(drain) += std::get<0>(drains[index]);
        std::get<1>(drain) += std::get<1>(drains[index]);
    }
    std::cerr << "\n";

    return drain;
}

template <typename T>
std::set<T> Union(std::vector<std::set<T>>&& vec) {
    while (vec.size() > 1) {
        std::vector<std::set<T>> reduction(vec.size() / 2);

        std::mutex mutex; 
        std::vector<std::thread> threads;
        for (size_t index = 0; index < reduction.size(); ++index) {
            threads.emplace_back([index, &mutex, &reduction, &vec] {
                std::set<T> uni = std::move(vec[index * 2]);
                if (index * 2 + 1 < vec.size()) {
                    std::set<T> ins = std::move(vec[index * 2 + 1]);
                    uni.insert(ins.begin(), ins.end());
                }

                std::lock_guard guard{mutex};
                reduction[index] = std::move(uni);
            });
        }
    
        for (auto &&thread : threads) {
            thread.join();
        }

        vec = std::move(reduction);
    }

    return vec.front();
}

std::tuple<size_t, size_t> Exact(const eg::MatrixXf &X, const std::vector<Class> &y, size_t k,
                                 size_t parallel) {
    size_t length = X.rows();
    size_t dim = X.cols();

    size_t truenum = 0, falsenum = 0;
    for (auto a : y) {
        if (a == Class::Positive) {
            truenum++;
        } else if (a == Class::Negative) {
            falsenum++;
        }
    }

    // build classifiers
    std::vector<LinearClassifier> clfs;
    {
        auto combs = Combs(length, dim);
        do {
            auto comb = combs.Get();
            clfs.emplace_back(X(comb, eg::indexing::all));
        } while (combs.Next());
    }

    std::vector<std::set<Mask>> drains(parallel);

    std::mutex mutex;
    std::vector<std::thread> threads;
    for (size_t index = 0; index < parallel; ++index) {
        threads.emplace_back([index, parallel, &mutex, &drains, &X, &y, k, &clfs] {
            auto begin = clfs.size() * index / parallel;
            auto end = clfs.size() * (index + 1) / parallel;

            std::set<Mask> colors;

            if (index == parallel - 1) {
                colors = Exact<true>(X, y, k, {clfs.begin() + begin, clfs.begin() + end});
            } else {
                colors = Exact<false>(X, y, k, {clfs.begin() + begin, clfs.begin() + end});
            }

            std::lock_guard guard{mutex};
            drains[index] = std::move(colors);
        });
    }

    for (auto &&thread : threads) {
        thread.join();
    }

    std::set<Mask> drain;
    for (auto index : tq::trange(parallel)) {
        drain.insert(drains[index].begin(), drains[index].end());
    }
    std::cerr << "\n";

    return {drain.size(), kBinom};
}
