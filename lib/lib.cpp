#include <iostream>

#include <iterator>
#include <functional>
#include <optional>
#include <random>

#include <tuple>
#include <vector>
#include <set>
#include <map>

#include <cmath>
#include <algorithm>

#include <Eigen/Dense>

// #include <pybind11/pybind11.h>
// #include <pybind11/numpy.h>
// #include <pybind11/stl.h>

// namespace py = pybind11;
namespace eg = Eigen;

const int kSeed = 733;

const float kEpsilon = 1e-5;

const int kPositive = 1;
const int kNegative = 0;
const int kZero = -1;

// techincal
template <class Iter, class Stream>
void Print(Iter begin, Iter end, Stream *stream) {
    for (auto iter = begin; iter != end; ++iter) {
        *stream << *iter << " ";
    }

    *stream << "\n";
}

template <class Iter, class T>
void Combinations(Iter begin, Iter end, size_t size, std::vector<T> &cmb,
                  std::vector<std::vector<T>> &dmp) {
    if (!size) {
        dmp.push_back(cmb);
        return;
    }

    for (auto iter = begin; iter <= end - size; ++iter) {
        cmb[size - 1] = *iter;
        Combinations(iter + 1, end, size - 1, cmb, dmp);
    }
}

template <class Iter, class T>
auto Combinations(Iter begin, Iter end, size_t size) {
    if (size > std::distance(begin, end)) {
        return std::vector<std::vector<T>>{};
    }

    std::vector<std::vector<T>> dmp;
    Combinations(begin, end, size, std::vector<T>{size}, dmp);

    return dmp;
}

auto Combinations(size_t end, size_t size) {
    std::vector<size_t> indexes{end};
    for (size_t index = 0; index < end; ++index) {
        indexes[index] = index;
    }

    return Combinations<std::vector<size_t>::iterator, size_t>(indexes.begin(), indexes.end(),
                                                               size);
}

template <class Iter, class T>
auto Compliment(Iter begin, Iter end, const std::vector<std::vector<T>> &cmbs) {
    std::vector<std::vector<T>> cmps;
    for (const auto& cmb : cmbs) {
        auto set = std::set{cmb.begin(), cmb.end()};

        std::vector<T> cmp;
        for (auto val : cmb) {
            if (set.find(val) == set.end()) {
                cmp.emplace_back(std::move(val));
            }
        }

        cmps.emplace_back(std::move(cmp));
    }

    return cmps;
}

std::vector<int> Reverse(const std::vector<int> &y) {
    std::vector<int> reverse{y};

    for (auto &c : reverse) {
        if (c == kPositive) {
            c = kNegative;
        } else if (c == kNegative) {
            c = kPositive;
        }
    }

    return reverse;
}
// techincal

// metrics
struct Confusion {
    Confusion(const std::vector<int> &ground, const std::vector<int> &pred) {
        for (size_t i = 0; i < ground.size(); ++i) {
            if (pred[i] == kZero) {
                un++;
            }

            if (pred[i] == kPositive && ground[i] == kPositive) {
                tp++;
            }

            if (pred[i] == kPositive && ground[i] == kNegative) {
                fp++;
            }

            if (pred[i] == kNegative && ground[i] == kPositive) {
                fn++;
            }

            if (pred[i] == kNegative && ground[i] == kNegative) {
                tn++;
            }
        }
    }

    size_t Error() {
        return fp + fn;
    }

    size_t Accur() {
        return tp + tn + un;
    }

    size_t tp = 0;
    size_t fp = 0;
    size_t fn = 0;
    size_t tn = 0;
    size_t un = 0;
};
// metrics

// normalization
eg::VectorXf Normalize(const eg::MatrixXf &X) {
    size_t dim = X.cols();

    const auto &point = X(0, eg::indexing::all);
    auto matrix = X.rowwise() - point.transpose();

    eg::BDCSVD<eg::MatrixXf, eg::ComputeFullV> svd{matrix};

    eg::VectorXf vec = eg::VectorXf::Zero(dim);
    vec(dim - 1) = 1.;

    return svd.matrixV() * vec;
}
// normalization

// linear classifier
struct LinearClassifier {
    eg::VectorXf point;
    eg::VectorXf normal;

    std::vector<int> Predict(const eg::MatrixXf &X) const {
        size_t length = X.rows();

        std::vector<int> pred;
        pred.assign(length, 0);

        auto product = (X.rowwise() - point.transpose()) * normal;

        for (size_t i = 0; i < length; ++i) {
            if (product(i) > kEpsilon) {
                pred[i] = kPositive;
            } else if (product(i) < -kEpsilon) {
                pred[i] = kNegative;
            } else {
                pred[i] = kZero;
            }
        }

        return pred;
    }

    static std::vector<LinearClassifier> Build(
        const eg::MatrixXf &X, const std::vector<std::vector<size_t>> &combinations) {
        std::vector<LinearClassifier> classifiers;
        for (const auto &cmb : combinations) {
            auto point = X(cmb.front(), eg::indexing::all);
            auto normal = Normalize(X(cmb, eg::indexing::all));

            classifiers.push_back({point, normal});
        }

        return classifiers;
    }

    static std::vector<LinearClassifier> Build(const eg::MatrixXf &X) {
        size_t length = X.rows();
        size_t dim = X.cols();

        auto combinations = Combinations(length, dim);

        return LinearClassifier::Build(X, combinations);
    }

    static LinearClassifier Best(const eg::MatrixXf &X,
                                 const std::vector<LinearClassifier> &classifiers,
                                 const std::vector<size_t> &indexes,
                                 std::function<size_t(const std::vector<int> &)> Score) {
        size_t length = X.rows();
        size_t dim = X.cols();

        auto best = classifiers.front();
        auto best_score = Score(best.Predict(X));

        for (auto index : indexes) {
            auto clf = classifiers[index];
            auto pred = clf.Predict(X);

            for (auto reverse : {false, true}) {
                if (reverse) {
                    pred = Reverse(pred);
                    clf.normal = -clf.normal;
                }

                auto score = Score(pred);
                if (score < best_score) {
                    best = clf;
                    best_score = score;
                }
            }
        }

        return best;
    }

    static LinearClassifier Best(const eg::MatrixXf &X,
                                 const std::vector<LinearClassifier> &classifiers,
                                 std::function<size_t(const std::vector<int> &)> Score) {
        std::vector<size_t> indexes{classifiers.size()};
        for (size_t index = 0; index < indexes.size(); ++index) {
            indexes[index] = index;
        }

        return LinearClassifier::Best(X, classifiers, indexes, Score);
    }
};
// linear classifier

std::tuple<size_t, size_t> Approximate(const eg::MatrixXf &X, const std::vector<int> &y, size_t k,
                                       float eps) {
    size_t length = X.rows();
    size_t dim = X.cols();

    // classifiers building
    std::map<std::vector<size_t>, size_t> map;
    std::vector<LinearClassifier> classifiers;
    {
        auto combinations = Combinations(length, dim);
        for (size_t index = 0; index < combinations.size(); ++index) {
            map.insert({combinations[index], index});
        }
        classifiers = LinearClassifier::Build(X, combinations);
    }

    // iteration
    std::mt19937 rng(kSeed);

    size_t count = 0;
    size_t iters = static_cast<size_t>(1 / (eps * eps));
    for (size_t i = 0; i < iters; ++i) {
        std::shuffle(y.begin(), y.end(), rng);

        std::vector<size_t> trueinds, falseinds;
        for (size_t i = 0; i < length; ++i) {
            if (y[i] == kPositive) {
                trueinds.push_back(i);
            } else if (y[i] == kNegative) {
                falseinds.push_back(i);
            }
        }

        std::vector<size_t> indexes;
        for (const auto &cmb : Combinations<std::vector<size_t>::iterator, size_t>(
                 trueinds.begin(), trueinds.end(), dim)) {
            indexes.push_back(map[cmb]);
        }
        for (const auto &cmb : Combinations<std::vector<size_t>::iterator, size_t>(
                 falseinds.begin(), falseinds.end(), dim)) {
            indexes.push_back(map[cmb]);
        }

        auto best =
            LinearClassifier::Best(X, classifiers, indexes, [&y](const std::vector<int> &pred) {
                return Confusion{y, pred}.Error();
            });

        if (Confusion{y, best.Predict(X)}.Error() <= k) {
            count++;
        }
    }

    return {count, iters};
}

size_t Binom(size_t n, size_t k) {
    if (!k || !n) {
        return 1;
    }

    return Binom(n - 1, k - 1) + Binom(n - 1, k);
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

std::tuple<size_t, size_t> Exact(const eg::MatrixXf &X, const std::vector<int> &y, size_t k) {
    size_t length = X.rows();
    size_t dim = X.cols();

    size_t truenum = 0, falsenum = 0;
    for (auto a : y) {
        if (a == kPositive) {
            truenum++;
        } else if (a == kNegative) {
            falsenum++;
        }
    }

    // classifiers building
    std::map<std::vector<size_t>, size_t> map;
    std::vector<LinearClassifier> classifiers;
    {
        auto combinations = Combinations(length, dim);
        for (size_t index = 0; index < combinations.size(); ++index) {
            map.insert({combinations[index], index});
        }
        classifiers = LinearClassifier::Build(X, combinations);
    }

    // iteration
    std::set<std::vector<size_t>> colors;
    for (const auto &[cmb, index] : map) {
        auto clf = classifiers[index];
        auto pred = clf.Predict(X);

        for (auto reverse : {false, true}) {
            if (reverse) {
                pred = Reverse(pred);
                clf.normal = -clf.normal;
            }

            // predict
            std::vector<size_t> posinds, neginds, zeroinds;
            for (size_t i = 0; i < length; ++i) {
                if (pred[i] == kPositive) {
                    posinds.push_back(i);
                } else if (pred[i] == kNegative) {
                    neginds.push_back(i);
                } else {
                    zeroinds.push_back(i);
                }
            }

            // distribute zeros
            for (size_t zero2pos = 0; zero2pos <= zeroinds.size(); ++zero2pos) {
                auto zero2neg = zeroinds.size() - zero2pos;

                for (size_t err = 0; err <= k; ++err) {
                    auto distribute = Distribute(posinds.size() + zero2pos,
                                                 neginds.size() + zero2neg, truenum, falsenum, err);

                    if (!distribute) {
                        continue;
                    }

                    auto [tp, fp, fn, tn] = distribute.value();

                    auto combs2pos = Combinations<std::vector<size_t>::iterator, size_t>(
                        zeroinds.begin(), zeroinds.end(), zero2pos);
                    auto combs2neg = Compliment<std::vector<size_t>::iterator, size_t>(
                        zeroinds.begin(), zeroinds.end(), combs2pos);
                    for (size_t zcmbi = 0; zcmbi < combs2pos.size(); ++zcmbi) {
                        posinds.insert(posinds.end(), combs2pos[zcmbi].begin(), combs2pos[zcmbi].end());
                        neginds.insert(neginds.end(), combs2pos[zcmbi].begin(), combs2pos[zcmbi].end());
                    }

                    break;
                }
            }
        }
    }

    return {colors.size(), Binom(truenum + falsenum, truenum)};
}

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