#pragma once

#include "eigen/Eigen/Dense"

#include <vector>
#include <tuple>

namespace eg = Eigen;

// class
enum Class { Positive = 1, Negative = -1, Zero = 0 };

std::vector<Class> Reverse(const std::vector<Class> &y);
// class

// metrics
class Confusion {
public:
    Confusion(const std::vector<Class> &ground, const std::vector<Class> &pred);

    int Error() const;
    int Accur() const;

private:
    int tp_ = 0, fp_ = 0;
    int fn_ = 0, tn_ = 0;
    int un_ = 0;
};
// metrics

// normalization
eg::VectorXf Normalize(const eg::MatrixXf &X);
// normalization

// linclass
class LinearClassifier;
namespace std {
template <>
class hash<LinearClassifier> {
public:
    std::size_t operator()(const LinearClassifier &clf) const;
};
}  // namespace std

class LinearClassifier {
    friend class std::hash<LinearClassifier>;

public:
    LinearClassifier(const eg::VectorXf &point, const eg::VectorXf &normal);
    LinearClassifier(const eg::MatrixXf &X);

    static LinearClassifier Fit(const eg::MatrixXf &X,
                                std::function<int(const std::vector<Class> &)> Loss);
    static LinearClassifier Fit(const eg::MatrixXf &X,
                                std::function<int(const std::vector<Class> &)> Loss,
                                const std::vector<LinearClassifier> &clfs);
    std::vector<Class> Predict(const eg::MatrixXf &X) const;

    LinearClassifier Opposite() const;

    // bool operator==(const LinearClassifier& rhs) const;

private:
    eg::VectorXf point_;
    eg::VectorXf normal_;
};
// linclass

std::tuple<size_t, size_t> Approximate(const eg::MatrixXf &X, std::vector<Class> y, size_t k,
                                       float eps, size_t parallel);
std::tuple<size_t, size_t> Exact(const eg::MatrixXf &X, const std::vector<Class> &y, size_t k,
                                 size_t parallel);