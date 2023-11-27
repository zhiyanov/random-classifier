#include <random>
#include <vector>
#include <iostream>
#include <cmath>
#include <algorithm>

#include <Eigen/Dense>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>


namespace py = pybind11;
namespace eg = Eigen;

const int SEED = 733;
const float EPSILON = 1e-5;
const int POSITIVE = 1;
const int NEGATIVE = 0;


void print(const std::vector<int> &vec) {
    for (auto const &a: vec) {
        std::cout << a << " ";
    }
    std::cout << "\n";
}

void print(const std::vector<float> &vec) {
    for (auto const &a: vec) {
        std::cout << a << " ";
    }
    std::cout << "\n";
}

void _comb(
        const std::vector<int> &arr,
        int start, int size,
        std::vector<int> &res,
        std::vector<int> &dmp
) {
    if (size <= 0) {
        for (auto const &a: res) {
            dmp.push_back(a);
        }
    } else {
        for (int i = start; i <= arr.size() - size; ++i) {
            res[size - 1] = arr[i];
            _comb(arr, i + 1, size - 1, res, dmp);
        }
    }
}

std::vector<int> comb(
        const std::vector<int> &arr,
        int size
) {
    std::vector<int> combs;
    std::vector<int> res(size);
    _comb(arr, 0, size, res, combs);

    return combs;
}

int misclass(
        const eg::MatrixXf &X,
        const std::vector<int> &y,
        const eg::VectorXf &normal, 
        const eg::VectorXf &point
) {
    int tp = 0, fn = 0;
    int fp = 0, tn = 0;
    
    eg::VectorXf product = (X.rowwise() - point.transpose()) * normal;
    for (int i = 0; i < y.size(); ++i) {
        float prod = product(i);
        
        if ((prod > EPSILON) && (y[i] == POSITIVE)) {
            tp += 1;
        } else if ((prod > EPSILON) && (y[i] == NEGATIVE)) {
            fp += 1;
        } else if ((prod < -EPSILON) && (y[i] == POSITIVE)) {
            fn += 1;
        } else if ((prod < -EPSILON) && (y[i] == NEGATIVE)) {
            tn += 1;
        }
    }
    
    return fp + fn;
}

std::tuple<int, int, int> classify(
        const eg::MatrixXf &X,
        const eg::VectorXf &normal, 
        const eg::VectorXf &point
) {
    int length = X.rows();
    int dim = X.cols();

    int posnum = 0, negnum = 0, zernum = 0;

    for (int i = 0; i < length; ++i) {
        eg::VectorXf vector = X(i, eg::indexing::all) - point.transpose();
        float product = vector.transpose() * normal;
        
        if (product > EPSILON) {
            posnum += 1;
        } else if (product < -EPSILON) {
            negnum += 1;
        } else {
            zernum += 1;
        }
    }
    
    return std::tuple<int, int, int>({posnum, negnum, zernum});
}

eg::VectorXf normalize(
        const eg::MatrixXf &X,
        const std::vector<int> &cmb
) {
    int length = X.rows();
    int dim = X.cols();

    eg::VectorXf point = X(cmb[0], eg::indexing::all);
    eg::MatrixXf M = X.rowwise() - point.transpose();

    eg::BDCSVD<eg::MatrixXf, eg::ComputeFullV> svd(M(
            cmb,
            eg::indexing::all
    ));
    
    eg::VectorXf normal = eg::VectorXf::Zero(dim);
    normal(dim - 1) = 1;
    normal = svd.matrixV() * normal;
    
    return normal;
}

std::map<std::vector<int>, eg::VectorXf> normalize(
        const eg::MatrixXf &X
) {
    int length = X.rows();
    int dim = X.cols();

    std::vector<int> indexes(length);
    for (int i = 0; i < indexes.size(); ++i) {
        indexes[i] = i;
    }
    
    std::map<std::vector<int>, eg::VectorXf> result;
  
    std::vector<int> cmbs = comb(indexes, dim);
    for (int i = 0; i < cmbs.size() / dim; ++i) {
        std::vector<int> cmb = std::vector<int>(
                cmbs.begin() + i * dim,
                cmbs.begin() + (i + 1) * dim
        );
        
        eg::MatrixXf matrix = X;
        eg::VectorXf point = X(cmb.back(), eg::indexing::all);
        matrix.rowwise() -= point.transpose(); 

        eg::VectorXf normal = normalize(
                matrix,
                cmb
        );
        
        result.insert({cmb, normal});
    }
    
    return result;
}

int enm_classify(
        const eg::MatrixXf &X,
        const std::vector<int> &y,
        std::map<std::vector<int>, eg::VectorXf> &normals
) {
    int length = X.rows();
    int dim = X.cols();

    std::vector<int> posinds, neginds;
    for (int i = 0; i < length; ++i) {
        if (y[i] == POSITIVE) {
            posinds.push_back(i);
        } else if (y[i] == NEGATIVE) {
            neginds.push_back(i);
        }
    }

    if ((posinds.size() <= dim) || (neginds.size() <= dim)) {
        return 0;
    }

    std::vector<int> poscmbs = comb(posinds, dim);
    std::vector<int> negcmbs = comb(neginds, dim);

    std::vector<int> cmbs;
    cmbs.insert(cmbs.end(), poscmbs.begin(), poscmbs.end());
    cmbs.insert(cmbs.end(), negcmbs.begin(), negcmbs.end());

    int result = y.size();
    for (int i = 0; i < cmbs.size() / dim; ++i) {
        std::vector<int> cmb = std::vector<int>(
                cmbs.begin() + i * dim,
                cmbs.begin() + (i + 1) * dim
        );

        eg::VectorXf normal = normals[cmb];
        eg::VectorXf point = X(cmb[0], eg::indexing::all);
        
        for (const auto &sign: {1., -1.}) {
            int score = misclass(
                    X, y,
                    normal * sign, point
            );

            if (score < result) {
                result = score;
            }
        }
    }
    
    return result;
}

std::tuple<int, int> enm_proba_apprx(
    const py::array_t<float, py::array::c_style> &D,
    const py::array_t<int, py::array::c_style> &c,
    int k,
    float eps
) {
    std::mt19937 rng(SEED);

    py::buffer_info D_buf = D.request();
    int length = D_buf.shape[0];
    int dim = D_buf.shape[1];
    
    float* D_ptr = (float *) D.request().ptr;
    eg::MatrixXf X(length, dim);
    for (int i = 0; i < length * dim; ++i) {
        X(i / dim, i % dim) = *(D_ptr + i);
    }

    std::vector<int> y(
            (int *) c.request().ptr,
            (int *) c.request().ptr + length
    );

    std::map<std::vector<int>, eg::VectorXf> normals = \
            normalize(X);

    int count = 0;
    int iters = (int) (1 / (eps * eps));
    for (int i = 0; i < iters; ++i) {
        std::shuffle(y.begin(), y.end(), rng);

        int score = enm_classify(
                X, y, normals
        );
        
        if (score <= k) {
            count += 1;
        }
    }

    return std::tuple<int, int>({count, iters});
}

int binom(int n, int k) {
   if ((k == 0) || (k == n)) {
       return 1;
   }

   return binom(n - 1, k - 1) + binom(n - 1, k);
}

std::tuple<int, int> enm_proba_exact(
    const py::array_t<float, py::array::c_style> &D,
    const py::array_t<int, py::array::c_style> &c
) {
    std::mt19937 rng(SEED);

    py::buffer_info D_buf = D.request();
    int length = D_buf.shape[0];
    int dim = D_buf.shape[1];
    
    float* D_ptr = (float *) D.request().ptr;
    eg::MatrixXf X(length, dim);
    for (int i = 0; i < length * dim; ++i) {
        X(i / dim, i % dim) = *(D_ptr + i);
    }

    std::vector<int> y(
            (int *) c.request().ptr,
            (int *) c.request().ptr + length
    );

    int posnum = 0, negnum = 0;
    for (auto const &a: y) {
        if (a == POSITIVE) {
            posnum += 1;
        } else if (a == NEGATIVE) {
            negnum += 1;
        }
    }
    
    std::vector<int> indexes(length);
    for (int i = 0; i < indexes.size(); ++i) {
        indexes[i] = i;
    }
    
    std::set<std::vector<int>> colors;
    std::vector<int> cmbs = comb(indexes, dim);
    for (int i = 0; i < cmbs.size() / dim; ++i) {
        std::vector<int> cmb = std::vector<int>(
                cmbs.begin() + i * dim,
                cmbs.begin() + (i + 1) * dim
        );
        
        eg::VectorXf normal = normalize(X, cmb);
        eg::VectorXf point = X(cmb[0], eg::indexing::all);
        for (const auto &sign: {1., -1.}) {
            normal = normal * sign;

            std::tuple<int, int, int> clf = classify(
                    X, normal, point
            );
            int pos = std::get<0>(clf); 
            int neg = std::get<1>(clf); 
            int zer = std::get<2>(clf);

            int score = std::abs(posnum - pos) + \
                        std::abs(negnum - neg) - zer;

            if (score > 0) {
                continue;
            }

            eg::VectorXf product = (
                    X.rowwise() - point.transpose()
            ) * normal;

            std::vector<int> positives;
            std::vector<int> zeroes;
            for (int j = 0; j < length; ++j) {
                float prod = product(j);

                if (prod > EPSILON) {
                    positives.push_back(j);
                } else if (std::abs(prod) < EPSILON) {
                    zeroes.push_back(j);
                }
            }
            
            int diff = posnum - pos;

            if (diff == 0) {
                std::sort(positives.begin(), positives.end());
                colors.insert(positives);
                continue;
            }

            std::vector<int> zcmbs = comb(zeroes, diff);
            for (int j = 0; j < zcmbs.size() / diff; ++j) {
                std::vector<int> points(positives);
                std::vector<int> zcmb = std::vector<int>(
                        zcmbs.begin() + j * diff,
                        zcmbs.begin() + (j + 1) * diff
                );
                
                for (auto const &a: zcmb) {
                    points.push_back(a);
                }

                std::sort(points.begin(), points.end());

                colors.insert(points);
            }
        }
    }

    int count = colors.size();
    return std::tuple<int, int>(
            {count, binom(posnum + negnum, posnum)}
    );
}

PYBIND11_MODULE(fast, m) {
    m.def("enm_proba_exact", &enm_proba_exact);
    m.def("enm_proba_apprx", &enm_proba_apprx);
}
