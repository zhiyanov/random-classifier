#include "classification.h"
#include "combinatorics.h"

#include "eigen/Eigen/Dense"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <vector>

namespace eg = Eigen;
namespace py = pybind11;

constexpr int kPositive = 1;
constexpr int kNegative = 0;

std::tuple<eg::MatrixXf, std::vector<Class>> Extract(const py::array_t<float, py::array::c_style> &D,
                                                     const py::array_t<int, py::array::c_style> &c) {
    size_t length = D.request().shape[0];
    size_t dim = D.request().shape[1];

    // data extraction
    auto *data_ptr = static_cast<float *>(D.request().ptr);
    eg::MatrixXf X{length, dim};
    for (size_t index = 0; index < length * dim; ++index) {
        X(index / dim, index % dim) = *(data_ptr + index);
    }
    
    // class extraction
    auto *class_ptr = static_cast<float *>(c.request().ptr);
    std::vector<Class> y;
    for (size_t index  = 0; index < length; ++index) {
        if (*class_ptr == kPositive) {
            y.push_back(Class::Positive);
        } else if (*class_ptr == kNegative) {
            y.push_back(Class::Negative);
        }
    }

    return {X, y};
}

std::tuple<size_t, size_t> enm_proba_exact(const py::array_t<float, py::array::c_style> &D,
                                           const py::array_t<int, py::array::c_style> &c,
                                           size_t k) {
    auto [X, y] = Extract(D, c);
    return Exact(X, y, k);
}

std::tuple<size_t, size_t> enm_proba_apprx(const py::array_t<float, py::array::c_style> &D,
                                           const py::array_t<int, py::array::c_style> &c,
                                           size_t k, float eps) {
    auto [X, y] = Extract(D, c);
    return Approximate(X, y, k, eps);
}

PYBIND11_MODULE(fast, fast_) {
    fast_.def("enm_proba_exact", &enm_proba_exact);
    fast_.def("enm_proba_apprx", &enm_proba_apprx);
}