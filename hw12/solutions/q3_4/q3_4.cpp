#include "svm.h"
#include <armadillo>

struct linear_classifier_a : public separator_gen<linear_classifier_a> {

    linear_classifier_a(const arma::mat& w, double b) : w(w), b(b) {}
    linear_classifier_a(arma::mat&& w, double b) : w(w), b(b) {}

    inline double classify(arma::mat const& x) const {
        return arma::sign((w.t() * x).eval()(0, 0) + b);
    }

    inline arma::mat const& transform(const arma::mat& x) const {
        return x;
    }
    const arma::mat w;
    const double b;
};

struct non_linear_classifier_b : public separator_gen<non_linear_classifier_b> {

    non_linear_classifier_b(const arma::mat& w, double b) : w(w), b(b) {}
    non_linear_classifier_b(arma::mat&& w, double b) : w(w), b(b) {}

    inline double classify(arma::mat const& x) const {
        return arma::sign((w.t() * x).eval()(0, 0) + b);
    }

    inline arma::mat transform(const arma::mat& x) const {
        return arma::mat({x(0, 0) * x(0, 0) * x(0, 0) - x(0, 1), 
            x(0, 0) * x(0, 1)});
    }
    const arma::mat w;
    const double b;
};

void q3() {
    arma::mat const Data({{-1, 1, 1},{1, 0, 1},{-1, 0, -1}, {1, -1, 0}});
    linear_classifier_a ca(arma::mat({1, 0}), 0.0);   
    auto const b_a = ca.decision_boundary_2D(Data, 500);
    b_a.save("boundary_a.txt", arma::file_type::csv_ascii);
    non_linear_classifier_b cb(arma::mat({1, 0}), 0.0);
    auto const b_b = cb.decision_boundary_2D(Data, 500);
    b_b.save("boundary_b.txt", arma::file_type::csv_ascii);
}

void q4() {
    
}

int main() {
    // q3();
    return 0;
}