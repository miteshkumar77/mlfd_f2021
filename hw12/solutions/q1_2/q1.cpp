#include <armadillo> 
#include <iostream> 
#include <array>
#include <functional>

using transform_t = std::function<arma::mat(arma::mat const&)>;

template <std::size_t L>
using dimension_t = std::array<int, L+1>;

template <std::size_t L>
using neural_t = std::array<arma::mat, L+1>;

inline void augment_1(arma::mat & m) {
    m.insert_rows(0, arma::ones(1, m.n_cols));
}

inline void deaugment_1(arma::mat & m) {
    m.shed_row(0);
}

template <std::size_t L>
void print_layers(neural_t<L> const& W, std::string const& name) {
    std::cout << name << std::endl;
    for (auto const& w : W) {
        w.print("layer: ");
    }
}

template <std::size_t L>
neural_t<L> create_rand_nn(dimension_t<L> const& dims, double scale) {
    neural_t<L> W({});
    for (int l = 1; l <= L; ++l) {
        W[l] = arma::randn(dims[l-1]+1, dims[l]) * scale;
    }
    return W;
}

template <std::size_t L>
neural_t<L> create_dfl_nn(dimension_t<L> const& dims, double dfl) {
    neural_t<L> W({});
    for (int l = 1; l <= L; ++l) {
        W[l] = arma::mat(dims[l-1]+1, dims[l]);
        W[l].fill(dfl);
    }
    return W;
}

template <std::size_t L>
void fprop_nn(neural_t<L> const& W, arma::mat const& x, transform_t OutputT, neural_t<L>& X, neural_t<L>& S) {
    X[0] = arma::mat(x);
    for (int l = 1; l <= L; ++l) {
        S[l] = W[l].t() * X[l-1];
        X[l] = arma::tanh(S[l]);
        augment_1(X[l]);
    }
    S[L] = W[L].t() * X[L-1];
    X[L] = OutputT(S[L]);
}


template <std::size_t L>
inline void bprop_nn(neural_t<L> const& W, neural_t<L> const& X,
        arma::mat const& y, neural_t<L> & D) {
    for (int l = L-1; l > 0; --l) {
        arma::mat Tpl = (1 - X[l] % X[l]);
        deaugment_1(Tpl);
        arma::mat t1 = W[l+1] * D[l+1];
        deaugment_1(t1);
        D[l] = Tpl % t1;
    }
}

template <std::size_t L>
void bprop_nn_tanh(neural_t<L> const& W, neural_t<L> const& X,
        arma::mat const& y, neural_t<L> & D) {
    D[L] =  ((double)2.0) * (X[L] - y) * (1 - X[L] % X[L]);
    bprop_nn<L>(W, X, y, D);
}

template <std::size_t L>
void bprop_nn_identity(neural_t<L> const& W, neural_t<L> const&X,
        arma::mat const& y, neural_t<L>& D) {
    D[L] = ((double)2.0) * (X[L] - y);
    bprop_nn<L>(W, X, y, D);
}

inline arma::mat take_identity(arma::mat const& m) {
    return arma::mat(m);
}

inline arma::mat take_tanh(arma::mat const& m) {
    return arma::tanh(m);
}

template <std::size_t L>
arma::mat error(neural_t<L> const& W, transform_t OutputT,
        arma::mat const& Data) {
    
    arma::mat err = arma::zeros(1, 1);
    double N = Data.n_rows;
    for (int row_idx = 0; row_idx < Data.n_rows; ++row_idx) {
        auto x = Data.submat(row_idx, 0, row_idx, Data.n_cols-2).t();
        auto y = Data.submat(row_idx, Data.n_cols-1, row_idx, Data.n_cols-1);
        neural_t<L> X;
        neural_t<L> S;
        fprop_nn<L>(W, x, OutputT, X, S);
        err += arma::square(X[L] - y)/N;
    }
    return err;
}



template <std::size_t L>
arma::mat error_gradient_identity_nn(neural_t<L> const& W, arma::mat const& Data,
        neural_t<L> & G) {
    arma::mat Ein = arma::zeros(1, 1);
    neural_t<L> X;
    neural_t<L> S;
    neural_t<L> D;
    double N = Data.n_rows;
    for (int row_idx = 0; row_idx < Data.n_rows; ++row_idx) {
        auto x = Data.submat(row_idx, 0, row_idx, Data.n_cols-2).t();
        auto y = Data.submat(row_idx, Data.n_cols-1, row_idx, Data.n_cols-1);
        fprop_nn<L>(W, x, take_identity, X, S);
        bprop_nn_identity<L>(W, X, y, D);
        Ein += arma::pow(X[L] - y(0,0), 2)/N;
        for (int l = 1; l <= L; ++l) {
            G[l] += (X[l-1] * D[l].t())/N;
        }
    }
    return Ein;
}

template <std::size_t L> 
void init_grad(neural_t<L> const& W, neural_t<L>& G) {
    G[0] = arma::mat(0, 0);
    for (int l = 1; l <= L; ++l)
        G[l] = arma::zeros(W[l].n_rows, W[l].n_cols);
}

template <std::size_t L>
arma::mat error_gradient_tanh_nn(neural_t<L> const& W, arma::mat const& Data,
        neural_t<L> & G) {
    arma::mat Ein = arma::zeros(1, 1);
    neural_t<L> X;
    neural_t<L> S;
    neural_t<L> D;
    double N = Data.n_rows;
    for (int row_idx = 0; row_idx < Data.n_rows; ++row_idx) {
        auto x = Data.submat(row_idx, 0, row_idx, Data.n_cols-2).t();
        auto y = Data.submat(row_idx, Data.n_cols-1, row_idx, Data.n_cols-1);
        fprop_nn<L>(W, x, take_tanh, X, S);
        bprop_nn_tanh<L>(W, X, y, D);
        Ein += arma::pow(X[L] - y(0,0), 2)/N;
        for (int l = 1; l <= L; ++l) {
            G[l] += (X[l-1] * D[l].t()/N);
        }
    }
    return Ein;
}

template <std::size_t L>
void numerical_gradient_nn(neural_t<L> const& W, transform_t OutputT,
        arma::mat const& Data, double delta, neural_t<L> & G) {
    neural_t<L> W0(W);
    for (int l = 1; l <= L; ++l) {
        G[l] = arma::zeros(W[l].n_rows, W[l].n_cols);
        for (arma::uword r = 0; r < W[l].n_rows; ++r) {
            for (arma::uword c = 0; c < W[l].n_cols; ++c) {
                auto w0 = W0[l](r, c);
                W0[l](r,c) = w0 + delta/2.0;
                auto e1 = error<L>(W0, OutputT, Data);
                W0[l](r,c) = w0 - delta/2.0;
                auto e0 = error<L>(W0, OutputT, Data);
                G[l](r,c) = ((e1 - e0)/delta).eval()(0,0);
            }
        }
    }
}

template <std::size_t L>
double classification_error(neural_t<L> const& W, arma::mat const& Data) {
    neural_t<L> X;
    neural_t<L> S;
    double ans = 0;
    for (int i = 0; i < Data.n_rows; ++i) {
        fprop_nn<L>(W, Data.submat(i, 0, i, Data.n_cols-2).t(), take_identity, X, S);
        if (Data(i, Data.n_cols-1) != arma::sign(X[L](0,0))) {
            ans += 1;
        }
    }
    return ans/((double)Data.n_rows);
}

template <std::size_t L>
void variable_learning_rate_gd_tanh_nn(neural_t<L> const& W, arma::mat const& Data, double n0, 
        double alpha, double beta, size_t iterations,
        neural_t<L> & Wr, arma::mat& Einr) {

    Einr = arma::mat(iterations+1, 1);
    auto Ein_tmp = arma::mat(1, 1);

    Wr = neural_t<L>(W);
    neural_t<L> W_tmp(W);

    neural_t<L> G;
    neural_t<L> G_tmp;
    init_grad<L>(W, G);
    init_grad<L>(W, G_tmp);

    Einr(0, 0) = error_gradient_tanh_nn<L>(W, Data, G)(0, 0);
    // auto indices = arma::randi(iterations+1, 1, arma::distr_param(0, Data.n_rows-1));
    for (size_t i = 1; i <= iterations; ++i) {
        for (int l = 1; l <= L; ++l) {
            W_tmp[l] = Wr[l] - n0 * G[l];
        }
        // Ein_tmp = error_gradient_tanh_nn<L>(W_tmp, Data.submat(indices(i,0), 0, indices(i,0), Data.n_cols-1), G_tmp);
        Ein_tmp = error_gradient_tanh_nn<L>(W_tmp, Data, G_tmp);
        if (Ein_tmp(0, 0) < Einr(i-1, 0)) {
            Einr(i, 0) = Ein_tmp(0, 0);
            G.swap(G_tmp);
            Wr.swap(W_tmp);
            n0 *= alpha;
        } else {
            Einr(i, 0) = Einr(i-1, 0);
            n0 *= beta;
        }
        if (i % 10000 == 0) {
            std::cout << "iters done: " << i << " / " << iterations  << " n0: " << n0 << std::endl;
        }
    }
}

template <std::size_t L>
void variable_learning_rate_gd_identity_nn(neural_t<L> const& W, arma::mat const& Data, double n0, 
        double alpha, double beta, size_t iterations,
        neural_t<L> & Wr, arma::mat& Einr) {

    Einr = arma::mat(iterations+1, 1);
    auto Ein_tmp = arma::mat(1, 1);

    Wr = neural_t<L>(W);
    neural_t<L> W_tmp(W);

    neural_t<L> G;
    neural_t<L> G_tmp;
    init_grad<L>(W, G);
    init_grad<L>(W, G_tmp);

    Einr(0, 0) = error_gradient_identity_nn<L>(W, Data, G)(0, 0);

    for (size_t i = 1; i <= iterations; ++i) {
        for (int l = 1; l <= L; ++l) {
            W_tmp[l] = Wr[l] - n0 * G[l];
        }

        Ein_tmp = error_gradient_identity_nn<L>(W_tmp, Data, G_tmp);
        if (Ein_tmp(0, 0) < Einr(i-1, 0)) {
            Einr(i, 0) = Ein_tmp(0, 0);
            G.swap(G_tmp);
            Wr.swap(W_tmp);
            n0 *= alpha;
        } else {
            Einr(i, 0) = Einr(i-1, 0);
            n0 *= beta;
        }
        if (i % 10000 == 0) {
            std::cout << "iters done: " << i << " / " << iterations  << " n0: " << n0 << std::endl;
        }
    }
}


template <std::size_t L>
void variable_learning_rate_early_stop_gd_identity_nn(neural_t<L> const& W, arma::mat const& Data, double n0, 
        double alpha, double beta, size_t iterations,
        neural_t<L> & Wr, arma::mat& Einr, size_t num_train) {
    
    auto const train_data = Data.submat(0, 0, num_train-1, Data.n_cols-1);
    auto const val_data = Data.submat(num_train, 0, Data.n_rows-1, Data.n_cols-1);
    Einr = arma::mat(iterations+1, 1);
    auto Ein_tmp = arma::mat(1, 1);

    Wr = neural_t<L>(W);
    neural_t<L> W_tmp(W);

    neural_t<L> G;
    neural_t<L> G_tmp;
    init_grad<L>(W, G);
    init_grad<L>(W, G_tmp);
    Einr(0, 0) = error_gradient_identity_nn<L>(W, train_data, G)(0, 0);

    for (size_t i = 1; i <= iterations; ++i) {
        for (int l = 1; l <= L; ++l) {
            W_tmp[l] = Wr[l] - n0 * G[l];
        }

        Ein_tmp = error_gradient_identity_nn<L>(W_tmp, train_data, G_tmp);
        if (Ein_tmp(0, 0) < Einr(i-1, 0)) {
            Einr(i, 0) = Ein_tmp(0, 0);
            G.swap(G_tmp);
            Wr.swap(W_tmp);
            n0 *= alpha;
        } else {
            Einr(i, 0) = Einr(i-1, 0);
            n0 *= beta;
        }
        if (i % 10000 == 0) {
            std::cout << "iters done: " << i << " / " << iterations  << " n0: " << n0 << std::endl;
        }
    }
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

int main() { 
    constexpr std::size_t L = 2;
    dimension_t<L> dims{2, 10, 1};
    arma::mat TrainData;
    arma::mat TestData;
    TrainData.load("../f_train.txt", arma::file_type::csv_ascii);
    TestData.load("../f_test.txt", arma::file_type::csv_ascii);
    neural_t<L> W = create_rand_nn<L>(dims, compute_scale(TrainData, 0.1));
    // neural_t<L> W = create_dfl_nn<L>(dims, 0.25);
    neural_t<L> learned_W;
    print_layers<L>(W, "W before: ");
    arma::mat Einr;
    variable_learning_rate_gd_identity_nn<L>(W, TrainData, 0.01, 1.1, 0.9, 900, learned_W, Einr);
    print_layers<L>(learned_W, "W after: ");
    Einr.save("q2_a.txt", arma::file_type::csv_ascii);
    std::cout << "Classification Ein: " << classification_error<L>(learned_W, TrainData) << std::endl 
        << "Classification Eout: " << classification_error<L>(learned_W, TestData) << std::endl;

    // constexpr std::size_t L = 2;
    // dimension_t<L> dims{2, 2, 1};
    // neural_t<L> W = create_dfl_nn<L>(dims, 0.25);
    // arma::mat Data({1, 1, 2, 1});

    // neural_t<L> G_tanh;
    // init_grad<L>(W, G_tanh);

    // neural_t<L> G_identity;
    // init_grad<L>(W, G_identity);

    // neural_t<L> G_numeric_tanh;
    // init_grad<L>(W, G_numeric_tanh);

    // neural_t<L> G_numeric_identity;
    // init_grad<L>(W, G_numeric_identity);

    // auto Ein_tanh = error_gradient_tanh_nn<L>(W, Data, G_tanh);
    // numerical_gradient_nn<L>(W, take_tanh, Data, 0.0001, G_numeric_tanh);
    // auto Ein_identity = error_gradient_identity_nn<L>(W, Data, G_identity);
    // numerical_gradient_nn<L>(W, take_identity, Data, 0.0001, G_numeric_identity);
    // Ein_tanh.print("Ein_tanh: ");
    // print_layers<L>(G_tanh, "G_tanh: ");
    // print_layers<L>(G_numeric_tanh, "G_numeric_tanh: ");
    // Ein_identity.print("Ein_identity: ");
    // print_layers<L>(G_identity, "G_identity: ");
    // print_layers<L>(G_numeric_identity, "G_numeric_identity: ");
    
    return 0;
}