#ifndef HW12_NN_H
#define HW12_NN_H

#include <armadillo>
#include <iostream>
#include <algorithm>
#include <array>
#include <functional>

template<std::size_t L>
using neural_t = std::array<arma::mat, L+1>;
template<std::size_t L>
using dimension_t = std::array<int, L+1>;

inline void augment_1(arma::mat & m) {
    m.insert_rows(0, arma::ones(1, m.n_cols));
}

inline void deaugment_1(arma::mat & m) {
    m.shed_row(0);
}

inline arma::mat get_xn(arma::mat const& Data, size_t n) {
    return Data.submat(n, 0, n, Data.n_cols-2).t();
}

inline arma::mat get_yn(arma::mat const& Data, size_t n) {
    return Data.submat(n, Data.n_cols-1, n, Data.n_cols-1);
}

double compute_scale(arma::mat const& Data, double mult) {
    double res = LONG_MIN;
    for (int r = 0; r < Data.n_rows; ++r) {
        auto x = Data.submat(r, 0, r, Data.n_cols-2);
        res = std::max(res, (x.t() * x).eval()(0, 0));
    }
    return res * mult;
}

arma::mat linreg(arma::mat const& D) {
    auto X = D.submat(0, 0, D.n_rows-1, D.n_cols-1);
    auto Y = D.submat(0, D.n_cols-1, D.n_rows-1, D.n_cols-1);
    return (X.t() * X).i() * X.t() * Y;
}

template <typename OutT, std::size_t L>
class NeuralNetInterface {
public:

    arma::mat decision_boundary_2D(neural_t<L> const& W, arma::mat const& Data, size_t res) {
        auto x1s = Data.submat(0, 0, Data.n_rows-1, 0);
        auto x2s = Data.submat(0, 1, Data.n_rows-1, 1);
        double minx1 = x1s.min();
        double maxx1 = x1s.max();
        double minx2 = x2s.min();
        double maxx2 = x2s.max();
        auto x1_mesh = arma::linspace(minx1, maxx1, res);
        auto x2_mesh = arma::linspace(minx2, maxx2, res);
        arma::mat results(res, res);
        for (size_t i = 0; i < res; ++i) {
            for (size_t j = 0; j < res; ++j) {
                results(i, j) = classify(W, arma::mat({1, x1_mesh[i], x2_mesh[j]}).t());
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


    inline void print_layers(neural_t<L> const& W, std::string const& name) {
        std::cout << name << std::endl;
        for (auto const& w : W) {
            w.print("layer: ");
        }
    }

    inline neural_t<L> create_rand_nn(dimension_t<L> const& dims, double scale) {
        neural_t<L> W({});
        for (int l = 1; l <= L; ++l) {
            W[l] = arma::randn(dims[l-1]+1, dims[l]) * scale;
        }
        return W;
    }

    inline neural_t<L> create_dfl_nn(dimension_t<L> const& dims, double dfl) {
        neural_t<L> W({});
        for (int l = 1; l <= L; ++l) {
            W[l] = arma::mat(dims[l-1]+1, dims[l]);
            W[l].fill(dfl);
        }
        return W;
    }

    inline int classify(neural_t<L> const& W, arma::mat const& x) {
        neural_t<L> X({});
        neural_t<L> S({});
        fprop_nn(W, x, X, S);
        return arma::sign(X[L](0, 0));
    }

    inline double regression_error(neural_t<L> const& W, arma::mat const& Data) {
        OutT& u = static_cast<OutT&>(*this);
        double E = 0.0;
        double N = Data.n_rows;
        neural_t<L> X({});
        neural_t<L> S({});
        for (int n = 0; n < Data.n_rows; ++n) {
            auto x = get_xn(Data, n);
            auto y = get_yn(Data, n);
            fprop_nn(W, x, X, S);
            E += u.point_error(X[L](0, 0), y(0, 0));
            // E += arma::pow(X[L] - y, 2).eval()(0, 0);
        }
        return E;
    }

    inline double classification_error(neural_t<L> const& W, arma::mat const& Data) {
        double E = 0.0;
        double N = Data.n_rows;
        neural_t<L> X({});
        neural_t<L> S({});
        for (int n = 0; n < Data.n_rows; ++n) {
            auto x = get_xn(Data, n);
            auto y = get_yn(Data, n);
            fprop_nn(W, x, X, S);
            if (y(0, 0) != arma::sign(X[L](0, 0))) {
                E += 1.0;
            }
        }
        return E/N;
    }

    inline void fprop_nn(neural_t<L> const& W, arma::mat const& x,
            neural_t<L>& X, neural_t<L>& S) {
        X[0] = arma::mat(x);
        for (int l = 1; l <= L; ++l) {
            S[l] = W[l].t() * X[l-1];
            X[l] = arma::tanh(S[l]);
            augment_1(X[l]);
        }
        S[L] = W[L].t() * X[L-1];
        X[L] = static_cast<OutT&>(*this).output_transform(S[L]);
    }

    inline void bprop_nn(neural_t<L> const& W, neural_t<L> const& X,
            arma::mat const& y, neural_t<L>& D) {
        D[L] = static_cast<OutT&>(*this).inverse_output_transform(X[L], y);
        for (int l = L-1; l > 0; --l) {
            arma::mat Tpl = (1 - X[l] % X[l]);
            deaugment_1(Tpl);
            arma::mat t1 = W[l+1] * D[l+1];
            deaugment_1(t1);
            D[l] = Tpl % t1;
        }
    }

    inline void init_grad(neural_t<L> const& W, neural_t<L>& G) {
        G[0] = arma::mat(0,0);
        for (int l = 1; l <= L; ++l) {
            G[l] = arma::zeros(W[l].n_rows, W[l].n_cols);
        }
    }

    inline double error_gradient(neural_t<L> const& W, arma::mat const& Data, neural_t<L>& G) {
        OutT& u = static_cast<OutT&>(*this);
        arma::mat Ein = arma::zeros(1,1);
        neural_t<L> X({});
        neural_t<L> S({});
        neural_t<L> D({});
        zero_grad(G);
        double N = Data.n_rows;
        for (size_t n = 0; n < Data.n_rows; ++n) {
            auto xn = get_xn(Data, n);
            auto yn = get_yn(Data, n);
            fprop_nn(W, xn, X, S);
            bprop_nn(W, X, yn, D);
            Ein += u.point_error(X[L](0, 0), yn(0, 0));
            for (size_t l = 1; l <= L; ++l) {
                G[l] += (X[l-1] * D[l].t())/N;
            }
        }
        return Ein(0, 0);
    }

    inline void zero_grad(neural_t<L>& G) {
        for (auto & g : G) {
            g.fill(0.0);
        }
    }

    void variable_rate_gd(neural_t<L> const& W0, arma::mat const& Data,
            neural_t<L> & Wr, arma::mat & Einr, double nt, double alpha, double beta, size_t iterations) {

        neural_t<L> Gr;
        init_grad(W0, Gr);
        neural_t<L> G_tmp;
        init_grad(W0, G_tmp);

        neural_t<L> W_tmp(W0);
        Wr = neural_t<L>(W0);

        double Ein_tmp;

        Einr = arma::mat(iterations+1, 1);
        Einr(0, 0) = error_gradient(W0, Data, Gr);
        for (size_t n = 1; n <= iterations; ++n) {
            for (size_t l = 1; l <= L; ++l) {
                W_tmp[l] = Wr[l] - nt * Gr[l];
            }
            Ein_tmp = error_gradient(W_tmp, Data, G_tmp);
            if (Ein_tmp < Einr(n-1, 0)) {
                Einr(n, 0) = Ein_tmp;
                Gr.swap(G_tmp);
                Wr.swap(W_tmp);
                nt *= alpha;
            } else {
                Einr(n, 0) = Einr(n-1, 0);
                nt *= beta;
            }
            if (n % 10000 == 0) {
                std::cout << n << " / " << iterations << " completed, nt = " << nt << std::endl;
            }
        }
    }

    void variable_rate_early_stop_gd(neural_t<L> const& W0, arma::mat const& Data,
            neural_t<L> & Wr, arma::mat & Einr, arma::mat& Eval, double nt, double alpha, double beta, size_t iterations, size_t num_test) {
                
        auto const test_data = Data.submat(0, 0, num_test-1, Data.n_cols-1);
        auto const val_data = Data.submat(num_test, 0, Data.n_rows-1, Data.n_cols-1);

        neural_t<L> Gr;
        init_grad(W0, Gr);
        neural_t<L> G_tmp;
        init_grad(W0, G_tmp);

        neural_t<L> W_tmp(W0);
        neural_t<L> W_best(W0);

        Wr = neural_t<L>(W0);
        double Ein_tmp;

        Einr = arma::mat(iterations+1, 1);
        Einr(0, 0) = error_gradient(W0, test_data, Gr);

        Eval = arma::mat(iterations+1, 1);
        Eval(0, 0) = regression_error(W_best, val_data);
        
        double E_best = Eval(0, 0);
        for (size_t n = 1; n <= iterations; ++n) {
            for (size_t l = 1; l <= L; ++l) {
                W_tmp[l] = Wr[l] - nt * Gr[l];
            }
            Ein_tmp = error_gradient(W_tmp, test_data, G_tmp);
            if (Ein_tmp < Einr(n-1, 0)) {
                Einr(n, 0) = Ein_tmp;
                Gr.swap(G_tmp);
                Wr.swap(W_tmp);
                nt *= alpha;
                Eval(n, 0) = regression_error(Wr, val_data);
                if (Eval(n, 0) < E_best) {
                    E_best = Eval(n, 0);
                    W_best = neural_t<L>(Wr);
                }
            } else {
                Einr(n, 0) = Einr(n-1, 0);
                Eval(n, 0) = Eval(n-1, 0);
                nt *= beta;
            }
            if (n % 10000 == 0) {
                std::cout << n << " / " << iterations << " completed, nt = " << nt << std::endl;
            }
        }
        Wr.swap(W_best);
    }

    inline double aug_error_gradient(neural_t<L> const& W, arma::mat const& Data, neural_t<L>& G, double lambda) {
        arma::mat Ein = arma::zeros(1,1);
        neural_t<L> X({});
        neural_t<L> S({});
        neural_t<L> D({});
        zero_grad(G);
        double N = Data.n_rows;
        for (size_t n = 0; n < Data.n_rows; ++n) {
            auto xn = get_xn(Data, n);
            auto yn = get_yn(Data, n);
            fprop_nn(W, xn, X, S);
            bprop_nn(W, X, yn, D);
            double aug = std::accumulate(W.begin() + 1, W.end(), 0.0, [](double acc, arma::mat const& a) -> double {
                return acc + arma::accu((a % a));
            });
            Ein += (arma::pow(X[L] - yn, 2) + lambda * aug)/N;
            for (size_t l = 1; l <= L; ++l) {
                G[l] += (X[l-1] * D[l].t() + 2 * lambda * W[l])/N;
            }
        }
        return Ein(0, 0);
    }

    void variable_rate_reg_gd(neural_t<L> const& W0, arma::mat const& Data,
            neural_t<L> & Wr, arma::mat & Einr, double nt, double alpha, double beta, size_t iterations, double lambda) {

        neural_t<L> Gr;
        init_grad(W0, Gr);
        neural_t<L> G_tmp;
        init_grad(W0, G_tmp);

        neural_t<L> W_tmp(W0);
        Wr = neural_t<L>(W0);

        double Ein_tmp;

        Einr = arma::mat(iterations+1, 1);
        Einr(0, 0) = aug_error_gradient(W0, Data, Gr, lambda);
        for (size_t n = 1; n <= iterations; ++n) {
            for (size_t l = 1; l <= L; ++l) {
                W_tmp[l] = Wr[l] - nt * Gr[l];
            }
            Ein_tmp = aug_error_gradient(W_tmp, Data, G_tmp, lambda);
            if (Ein_tmp < Einr(n-1, 0)) {
                Einr(n, 0) = Ein_tmp;
                Gr.swap(G_tmp);
                Wr.swap(W_tmp);
                nt *= alpha;
            } else {
                Einr(n, 0) = Einr(n-1, 0);
                nt *= beta;
            }
            if (n % 10000 == 0) {
                std::cout << n << " / " << iterations << " completed, nt = " << nt << std::endl;
            }
        }
    }

    void numeric_gradient(neural_t<L> const& W, arma::mat const& Data, double delta, neural_t<L>& G) {
        init_grad(W, G);
        neural_t<L> Wc(W);
        for (size_t l = 1; l <= L; ++l) {
            for (size_t r = 0; r < W[l].n_rows; ++r) {
                for (size_t c = 0; c < W[l].n_cols; ++c) {
                    double original = Wc[l](r, c);
                    Wc[l](r, c) = original + delta/2.0;
                    auto Eplus = regression_error(Wc, Data);
                    Wc[l](r, c) = original - delta/2.0;
                    auto Eminus = regression_error(Wc, Data);
                    Wc[l][r, c] = original;
                    G[l](r, c) = (Eplus - Eminus)/delta;
                }
            }
        }
    }
};


template <std::size_t L>
class TanhNeuralNet : public NeuralNetInterface<TanhNeuralNet<L>, L> {
public:

    inline arma::mat output_transform(arma::mat const& m) const {
        return arma::tanh(m);
    }

    inline arma::mat inverse_output_transform(arma::mat const& XL, 
        arma::mat const& y) const {
        return ((double)1/2.0) * (XL - y) * (1 - XL % XL);
    }

    inline double point_error(double signal, double expected) const {
        return ((double)1/4.0) * ((signal - expected) * (signal - expected));
    }

    double scale=0.01;
};

template <std::size_t L>
class IdentityNeuralNet : public NeuralNetInterface<IdentityNeuralNet<L>, L> {
public:

    inline arma::mat inverse_output_transform(arma::mat const& XL, 
        arma::mat const& y) const {
        return ((double)1/2.0) * (XL - y);
    }

    inline arma::mat output_transform(arma::mat const& m) const {
        return arma::mat(m);
    }

    inline double point_error(double signal, double expected) const {
        return ((double)1/4.0) * ((signal - expected) * (signal - expected));
    }

    double scale=0.01;
};
#endif