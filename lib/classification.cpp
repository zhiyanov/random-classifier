#include "classification.h"
#include "combinatorics.h"

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

namespace eg = Eigen;

constexpr int kSeed = 733;
constexpr std::array kSeeds = {123, 452, 864, kSeed};

constexpr float kEpsilon = 1e-5;

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
LinearClassifier::LinearClassifier(const eg::VectorXf& point,
                                   const eg::VectorXf& normal)
    : point_{point}, normal_{normal} {
}

LinearClassifier::LinearClassifier(const eg::MatrixXf &X) {
    point_ = X(0, eg::indexing::all);
    normal_ = Normalize(X);
}

LinearClassifier LinearClassifier::Fit(const eg::MatrixXf &X, std::function<int(const std::vector<Class> &)> Loss) {
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

LinearClassifier LinearClassifier::Fit(const eg::MatrixXf &X, std::function<int(const std::vector<Class> &)> Loss,
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

template <int seed>
std::tuple<size_t, size_t> Approximate(const eg::MatrixXf &X, std::vector<Class> y,
                                       size_t k, float eps) {
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
    
    if (seed == kSeed) {
        for (auto iter : tq::trange(++iters)) {
            std::shuffle(y.begin(), y.end(), rng);
        
            auto loss = [&y](const std::vector<Class> &pred) {
                auto conf = Confusion{y, pred};
                return conf.Error();
            };

            auto clf = LinearClassifier::Fit(X, loss, clfs);

            if (loss(clf.Predict(X)) <= k) {
                count++;
            }
        }
        
        std::cerr << "\n";
    } else {
        for (size_t iter = 0; iter < iters; ++iter) {
            std::shuffle(y.begin(), y.end(), rng);
        
            auto loss = [&y](const std::vector<Class> &pred) {
                auto conf = Confusion{y, pred};
                return conf.Error();
            };

            auto clf = LinearClassifier::Fit(X, loss, clfs);

            if (loss(clf.Predict(X)) <= k) {
                count++;
            }
        }
    }

    return {count, iters};
}

std::tuple<size_t, size_t> Approximate(const eg::MatrixXf &X, std::vector<Class> y,
                                       size_t k, float eps) {
    std::array<std::tuple<size_t, size_t>, 4> apps;
    std::vector<std::thread> threads;
    threads.emplace_back([&]() {
        apps[0] = Approximate<kSeeds[0]>(X, y, k, eps * std::sqrt(static_cast<float>(4)));
    });
    threads.emplace_back([&]() {
        apps[1] = Approximate<kSeeds[1]>(X, y, k, eps * std::sqrt(static_cast<float>(4)));
    });
    threads.emplace_back([&]() {
        apps[2] = Approximate<kSeeds[2]>(X, y, k, eps * std::sqrt(static_cast<float>(4)));
    });
    threads.emplace_back([&]() {
        apps[3] = Approximate<kSeeds[3]>(X, y, k, eps * std::sqrt(static_cast<float>(4)));
    });

    for (auto&& thread: threads) {
        thread.join();
    }

    auto nominator = std::get<0>(apps[0]) + std::get<0>(apps[1]) + std::get<0>(apps[1]) + std::get<0>(apps[3]);
    auto denominator = std::get<1>(apps[0]) + std::get<1>(apps[1]) + std::get<1>(apps[1]) + std::get<1>(apps[3]);

    return {nominator, denominator};
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

std::tuple<size_t, size_t> Exact(const eg::MatrixXf &X, const std::vector<Class> &y, size_t k) {
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

    // iteration
    std::set<std::vector<int>> colors;
    for (auto clf : tq::tqdm(clfs)) {
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
                for (size_t err = 0; err <= k; ++err) {
                    auto distribute = Distribute(posinds.size() + zero2pos,
                                                 neginds.size() + zero2neg, truenum, falsenum, err);

                    if (!distribute) {
                        continue;
                    }

                    auto [tp, fp, fn, tn] = distribute.value();

                    auto zero_combs = Combinations(zeroinds.begin(), zeroinds.end(), zero2pos);
                    do {
                        auto z2p_cmb = zero_combs.Get();
                        auto z2n_cmb = Residual(zeroinds.begin(), zeroinds.end(),
                                                z2p_cmb.begin(), z2p_cmb.end());

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

                                std::sort(trues.begin(), trues.end());
                                colors.emplace(std::move(trues));
                            } while (fn_combs.Next());
                        } while (tp_combs.Next());
                    } while (zero_combs.Next());
                }
            }
        }
    }

    std::cerr << "\n";
    return {colors.size(), Binom(truenum + falsenum, truenum)};
}

/*
// std::tuple<eg::MatrixXf, std::vector<int>>
// Extract(const py::array_t<float, py::array::c_style> &D,
//         const py::array_t<int, py::array::c_style> &c) {
//
//     size_t length = D.request().shape[0];
//     size_t dim = D.request().shape[1];
//
//     // data extraction
//     auto *data_ptr = static_cast<float *>(D.request().ptr);
//     eg::MatrixXf X{length, dim};
//     for (size_t i = 0; i < length * dim; ++i) {
//         X(i / dim, i % dim) = *(data_ptr + i);
//     }
//
//     auto *class_ptr = static_cast<float *>(c.request().ptr);
//     std::vector<int> y{class_ptr, class_ptr + length};
//
//     return {X, y};
// }

// PYBIND11_MODULE(fast, m) {
//     m.def("enm_proba_exact", &enm_proba_exact);
//     m.def("enm_proba_apprx", &enm_proba_apprx);
// }
*/