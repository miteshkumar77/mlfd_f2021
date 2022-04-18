#ifndef SVM_H
#define SVM_H

#include <armadillo>



template<typename C>
struct separator_gen {
    arma::mat decision_boundary_2D(arma::mat const& Data, size_t res) {
        C const& u = static_cast<C const&>(*this);

        auto x1s = Data.submat(0, 1, Data.n_rows-1, 1);
        auto x2s = Data.submat(0, 2, Data.n_rows-1, 2);
        double minx1 = x1s.min();
        double maxx1 = x1s.max();
        double minx2 = x2s.min();
        double maxx2 = x2s.max();
        auto x1_mesh = arma::linspace(minx1, maxx1, res);
        auto x2_mesh = arma::linspace(minx2, maxx2, res);
        arma::mat results(res, res);
        size_t total_neg = 0;
        size_t total_pos = 0;
        for (size_t i = 0; i < res; ++i) {
            for (size_t j = 0; j < res; ++j) {
                results(i, j) = 
                    u.classify(u.transform(arma::mat({x1_mesh[i], x2_mesh[j]})));
                if (results(i, j) < 0) {
                    ++total_neg;
                } else {
                    ++total_pos;
                }
            }
        }
        std::array<int, 5> dirs{0, 1, 0, -1, 0};
        arma::mat separator(res, res);
        size_t num_seps = 0;
        for (size_t i = 1; i+1 < res; ++i) {
            for (size_t j = 1; j+1 < res; ++j) {
                size_t num_pos = 0;
                size_t num_neg = 0;
                for (size_t x = 0; x < 4; ++x) {
                    if (results(i+dirs[x], j+dirs[x+1]) > 0) {
                        ++num_pos;
                    } else {
                        ++num_neg;
                    }
                }
                if (num_pos > 0 && num_neg > 0) {
                    num_seps += 1;
                    separator(i, j) = 1;
                } else {
                    separator(i, j) = -1;
                }
            }
        }
        arma::mat sep_pts(num_seps, 2);
        size_t k = 0;
        for (size_t i = 1; i + 1 < res; ++i) {
            for (size_t j = 1; j + 1 < res; ++j) {
                if (separator(i, j) > 0) {
                    sep_pts(k, 0) = (x1_mesh(i, 0) + x1_mesh(i+1, 0))/2.0;
                    sep_pts(k++, 1) = (x2_mesh(j, 0) + x2_mesh(j+1, 0))/2.0;
                }
            }
        }

        return sep_pts;
    }

};


// template <typename C>
// struct svm_interface {
//     void make_separable_classifier(arma::mat const& Data) {
//         C const& u = static_cast<C const&>(*this);
//         auto X = Data.submat(0, 0, Data.n_rows-1, Data.n_cols-2);
//         auto Y = Data.submat(0, Data.n_cols-1, Data.n_rows-1, Data.n_cols-1);

//         arma::mat Qd(Data.n_rows, Data.n_rows);
//         for (size_t i = 0; i < Data.n_rows; ++i) {
//             for (size_t j = 0; j < Data.n_rows; ++j) {
//                 Qd(i, j) =
//                     arma::accu(Y(i, 0) * Y(j, 0) * u.Kernel(X.submat(i, 0, i, Data.n_cols-1),  X.submat(j, 0, j, Data.n_cols-1)));
//             }
//         }
//         arma::mat Ad = arma::join_rows(arma::join_rows(Y.t(), -1 * Y.t()), arma::eye(Data.n_rows, Data.n_rows));
//         arma::mat minus(Data.n_rows, 1);
//         minus.fill(-1.0);
//         arma::mat zeros = arma::zeros(Data.n_rows+2, 0);
//         // a* = QP(Qd, minus, Ad, zeros)


//     }
// };

#endif