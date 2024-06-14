#include "combinatorics.h"
#include "classification.h"

#include "eigen/Eigen/Dense"

#include <iostream>
#include <numeric>
#include <vector>
#include <cmath>

constexpr size_t kApproximate = 8;
constexpr size_t kExact = 4;

template <class Iter, class Stream>
void Print(Iter begin, Iter end, Stream *stream) {
    for (auto iter = begin; iter != end; ++iter) {
        *stream << *iter << " ";
    }

    *stream << "\n";
}

void TestCombinations(size_t length, size_t size) {
    size_t count = 0;

    auto combs = Combs(length, size);
    do {
        auto comb = combs.Get();
        Print(comb.begin(), comb.end(), &std::cout);
        count++;
    } while (combs.Next());

    std::cout << count << "\n";
}

void TestPermutations(size_t length, size_t size) {
    size_t count = 0;

    auto perms = Perms(length, size);
    do {
        auto perm = perms.Get();
        Print(perm.begin(), perm.end(), &std::cout);
        count++;
    } while (perms.Next());

    std::cout << count << "\n";
}

void TestResiduals(size_t length, size_t size) {
    size_t count = 0;

    auto combs = Combs(length, size);
    do {
        auto comb = combs.Get();
        auto resi = Residual(combs.Data().begin(), combs.Data().end(), comb.begin(), comb.end());
        Print(resi.begin(), resi.end(), &std::cout);
        count++;
    } while (combs.Next());

    std::cout << count << "\n";
}

void TestLinearClassifier(const eg::MatrixXf &X, const std::vector<Class> &y) {
    auto loss = [&y](const std::vector<Class> &pred) {
        auto conf = Confusion{y, pred};
        return conf.Error();
    };

    auto clf = LinearClassifier::Fit(X, loss);
    auto pred = clf.Predict(X);

    size_t dim = X.cols();

    Print(pred.begin(), pred.end(), &std::cout);
    std::cout << loss(pred) << "\n";
}

void TestApproximate(const eg::MatrixXf &X, const std::vector<Class> &y, size_t k, float eps) {
    auto [nominator, denominator] = Approximate(X, y, k, eps, kApproximate);
    std::cout << nominator << " " << denominator << "\n";
}

void TestExact(const eg::MatrixXf &X, const std::vector<Class> &y, size_t k) {
    auto [nominator, denominator] = Exact(X, y, k, kExact);
    std::cout << nominator << " " << denominator << "\n";
}

void TestProba(const eg::MatrixXf &X, const std::vector<Class> &y, size_t k, float eps) {
    auto [app_nominator, app_denominator] = Approximate(X, y, k, eps, kApproximate);
    auto [ext_nominator, ext_denominator] = Exact(X, y, k, kExact);

    std::cout << "Approximate: " << app_nominator << " " << app_denominator << "\n";
    std::cout << "Exact: " << ext_nominator << " " << ext_denominator << "\n";
}

int main() {
    // std::cout << "Combinations\n";
    // TestCombinations(5, 3);

    // std::cout << "Residuals\n";
    // TestResiduals(5, 3);

    // std::cout << "Permutations\n";
    // TestPermutations(5, 3);

    // std::cout << "TestLinearClassifier\n";
    // TestLinearClassifier(eg::MatrixXf({{0}, {0}, {0},
    //                                    {1}, {1}, {1}}),
    //                      std::vector<Class>({Class::Negative, Class::Negative, Class::Positive,
    //                                          Class::Negative, Class::Positive,
    //                                          Class::Positive}));

    // TestLinearClassifier(eg::MatrixXf({{0, 0}, {0, 1},
    //                                    {1, 0}, {1, 1}}),
    //                      std::vector<Class>({Class::Negative, Class::Negative,
    //                                          Class::Positive, Class::Positive}));

    // TestLinearClassifier(eg::MatrixXf({{0, 0}, {0, 1}, {0.5, 0},
    //                                    {1, 0}, {1, 1}, {0.5, 1}}),
    //                      std::vector<Class>({Class::Negative, Class::Negative, Class::Negative,
    //                                          Class::Positive, Class::Positive,
    //                                          Class::Positive}));

    // TestLinearClassifier(eg::MatrixXf({{0, 0}, {0, 0.2}, {0, 0.4}, {0, 0.6}, {0, 0.8}, {0, 1},
    // {0.2, 1}, {0.4, 1}, {0.6, 1}, {0.8, 1}, {1, 1},
    //                                    {0, 0}, {0.2, 0}, {0.4, 0}, {0.6, 0}, {0.8, 0}, {1, 0},
    //                                    {1, 0.2}, {1, 0.4}, {1, 0.6}, {1, 0.8}, {1, 1}}),
    //                      std::vector<Class>({Class::Negative, Class::Negative, Class::Negative,
    //                      Class::Negative, Class::Negative, Class::Negative, Class::Negative,
    //                      Class::Negative, Class::Negative, Class::Negative, Class::Negative,
    //                                          Class::Positive, Class::Positive, Class::Positive,
    //                                          Class::Positive, Class::Positive, Class::Positive,
    //                                          Class::Positive, Class::Positive, Class::Positive,
    //                                          Class::Positive, Class::Positive}));

    // std::cout << "TestApproximateZero\n";
    // TestApproximate(eg::MatrixXf({{0},
    //                               {1}}),
    //                 std::vector<Class>({Class::Negative,
    //                                     Class::Positive}),
    //                 0, 1e-1);

    // TestApproximate(eg::MatrixXf({{0}, {0.2},
    //                               {0.8}, {1}}),
    //                 std::vector<Class>({Class::Negative, Class::Negative,
    //                                     Class::Positive, Class::Positive}),
    //                 0, 1e-2);

    // TestApproximate(eg::MatrixXf({{0, 1e-1}, {0.2, 1e-1},
    //                               {0.8, -1e-1}, {1, -1e-1}}),
    //                 std::vector<Class>({Class::Negative, Class::Negative,
    //                                     Class::Positive, Class::Positive}),
    //                 0, 1e-2);

    // TestApproximate(eg::MatrixXf({{0, 0}, {0, 1},
    //                               {1, 0}, {1, 1}}),
    //                 std::vector<Class>({Class::Negative, Class::Negative,
    //                                     Class::Positive, Class::Positive}),
    //                 0, 1e-2);

    // TestApproximate(eg::MatrixXf({{0, 0}, {0, 0.2}, {0, 0.4}, {0, 0.6}, {0, 0.8}, {0, 1}, {0.2,
    // 1}, {0.4, 1}, {0.6, 1}, {0.8, 1}, {1, 1},
    //                              {0, 0}, {0.2, 0}, {0.4, 0}, {0.6, 0}, {0.8, 0}, {1, 0}, {1,
    //                              0.2}, {1, 0.4}, {1, 0.6}, {1, 0.8}, {1, 1}}),
    //                 std::vector<Class>({Class::Negative, Class::Negative, Class::Negative,
    //                 Class::Negative, Class::Negative, Class::Negative, Class::Negative,
    //                 Class::Negative, Class::Negative, Class::Negative, Class::Negative,
    //                                     Class::Positive, Class::Positive, Class::Positive,
    //                                     Class::Positive, Class::Positive, Class::Positive,
    //                                     Class::Positive, Class::Positive, Class::Positive,
    //                                     Class::Positive, Class::Positive}),
    //                 0, 1e-1);

    // std::cout << "TestApproximateOne\n";
    // TestApproximate(eg::MatrixXf({{0, 0}, {0, 0.2}, {0, 0.4}, {0, 0.6}, {0, 0.8}, {0, 1}, {0.2,
    // 1}, {0.4, 1}, {0.6, 1}, {0.8, 1}, {1, 1},
    //                              {0, 0}, {0.2, 0}, {0.4, 0}, {0.6, 0}, {0.8, 0}, {1, 0}, {1,
    //                              0.2}, {1, 0.4}, {1, 0.6}, {1, 0.8}, {1, 1}}),
    //                 std::vector<Class>({Class::Negative, Class::Negative, Class::Negative,
    //                 Class::Negative, Class::Negative, Class::Negative, Class::Negative,
    //                 Class::Negative, Class::Negative, Class::Negative, Class::Negative,
    //                                     Class::Positive, Class::Positive, Class::Positive,
    //                                     Class::Positive, Class::Positive, Class::Positive,
    //                                     Class::Positive, Class::Positive, Class::Positive,
    //                                     Class::Positive, Class::Positive}),
    //                 5, 1e-2);

    // std::cout << "TestExactZero\n";
    // TestExact(eg::MatrixXf({{0},
    //                         {1}}),
    //           std::vector<Class>({Class::Negative,
    //                               Class::Positive}),
    //           0);

    // TestExact(eg::MatrixXf({{0}, {0.2},
    //                         {0.8}, {1}}),
    //           std::vector<Class>({Class::Negative, Class::Negative,
    //                               Class::Positive, Class::Positive}),
    //           0);

    // TestExact(eg::MatrixXf({{0, 1e-1}, {0.2, 1e-1},
    //                         {0.8, -1e-1}, {1, -1e-1}}),
    //           std::vector<Class>({Class::Negative, Class::Negative,
    //                               Class::Positive, Class::Positive}),
    //           0);

    // TestExact(eg::MatrixXf({{0, 0}, {0, 1},
    //                         {1, 0}, {1, 1}}),
    //           std::vector<Class>({Class::Negative, Class::Negative,
    //                               Class::Positive, Class::Positive}),
    //           0);

    // TestExact(eg::MatrixXf({{0, 0}, {0, 0.2}, {0, 0.4}, {0, 0.6}, {0, 0.8}, {0, 1}, {0.2, 1},
    // {0.4, 1}, {0.6, 1}, {0.8, 1}, {1, 1},
    //                         {0, 0}, {0.2, 0}, {0.4, 0}, {0.6, 0}, {0.8, 0}, {1, 0}, {1, 0.2}, {1,
    //                         0.4}, {1, 0.6}, {1, 0.8}, {1, 1}}),
    //           std::vector<Class>({Class::Negative, Class::Negative, Class::Negative,
    //           Class::Negative, Class::Negative, Class::Negative, Class::Negative,
    //           Class::Negative, Class::Negative, Class::Negative, Class::Negative,
    //                               Class::Positive, Class::Positive, Class::Positive,
    //                               Class::Positive, Class::Positive, Class::Positive,
    //                               Class::Positive, Class::Positive, Class::Positive,
    //                               Class::Positive, Class::Positive}),
    //           0);

    // std::cout << "TestExactOne\n";
    // TestExact(eg::MatrixXf({{0, 0}, {0, 0.2}, {0, 0.4}, {0, 0.6}, {0, 0.8}, {0, 1}, {0.2, 1},
    // {0.4, 1}, {0.6, 1}, {0.8, 1}, {1, 1},
    //                         {0, 0}, {0.2, 0}, {0.4, 0}, {0.6, 0}, {0.8, 0}, {1, 0}, {1, 0.2}, {1,
    //                         0.4}, {1, 0.6}, {1, 0.8}, {1, 1}}),
    //           std::vector<Class>({Class::Negative, Class::Negative, Class::Negative,
    //           Class::Negative, Class::Negative, Class::Negative, Class::Negative,
    //           Class::Negative, Class::Negative, Class::Negative, Class::Negative,
    //                               Class::Positive, Class::Positive, Class::Positive,
    //                               Class::Positive, Class::Positive, Class::Positive,
    //                               Class::Positive, Class::Positive, Class::Positive,
    //                               Class::Positive, Class::Positive}),
    //           5);

    std::cout << "TestProba\n";
    TestProba(
        eg::MatrixXf({{0, 0},   {0, 0.2}, {0, 0.4}, {0, 0.6}, {0, 0.8}, {0, 1},
                      {0.2, 1}, {0.4, 1}, {0.6, 1}, {0.8, 1}, {1, 1},   {0, 0},
                      {0.2, 0}, {0.4, 0}, {0.6, 0}, {0.8, 0}, {1, 0},   {1, 0.2},
                      {1, 0.4}, {1, 0.6}, {1, 0.8}, {1, 1}}),
        std::vector<Class>({Class::Negative, Class::Negative, Class::Negative, Class::Negative,
                            Class::Negative, Class::Negative, Class::Negative, Class::Negative,
                            Class::Negative, Class::Negative, Class::Negative, Class::Positive,
                            Class::Positive, Class::Positive, Class::Positive, Class::Positive,
                            Class::Positive, Class::Positive, Class::Positive, Class::Positive,
                            Class::Positive, Class::Positive}),
        5, 1e-2);

    TestProba(
        eg::MatrixXf({{-0.54827213, 0.05780018},
                      {-0.05645907, 0.726147},
                      {1.0522097, 0.1956428},
                      {-0.24426988, 0.92464584}}),
        std::vector<Class>({Class::Negative, Class::Negative, Class::Positive, Class::Positive}), 0,
        1e-2);
}