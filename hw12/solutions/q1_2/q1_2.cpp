#include "nn.h"

void tmp() {
    constexpr std::size_t L = 2;
    dimension_t<L> dims{2, 10, 1};
    TanhNeuralNet<L> tnn;
    auto W = tnn.create_rand_nn(dims, 0.001);
    neural_t<L> S({});
    neural_t<L> X({});
    neural_t<L> D({});
    arma::mat const x = arma::mat({1.0,1.0,2.0}).t();
    arma::mat const y({1.0});
    tnn.fprop_nn(W, x, X, S);
    tnn.print_layers(X, "X: ");
    tnn.print_layers(S, "S: ");
    tnn.bprop_nn(W, X, y, D);
    tnn.print_layers(D, "D: ");
}

void test_fprop_bprop() {
    constexpr std::size_t L = 3;
    dimension_t<L> dims{1, 2, 1, 1};

    TanhNeuralNet<L> tnn;
    neural_t<L> W({});
    W[1] = arma::mat({{0.1, 0.2}, {0.3, 0.4}});
    W[2] = arma::mat({0.2, 1, -3}).t();
    W[3] = arma::mat({1, 2}).t();
    tnn.print_layers(W, "W: ");
    neural_t<L> S({});
    neural_t<L> X({});
    neural_t<L> D({});
    arma::mat const x = arma::mat({1.0,2.0}).t();
    arma::mat const y({1.0});
    tnn.fprop_nn(W, x, X, S);
    tnn.print_layers(X, "X: ");
    tnn.print_layers(S, "S: ");
    tnn.bprop_nn(W, X, y, D);
    tnn.print_layers(D, "D: ");
}

void q1() {
    constexpr std::size_t L = 2;
    dimension_t<L> dims{2, 2, 1};
    TanhNeuralNet<L> tnn;
    auto W = tnn.create_dfl_nn(dims, 0.25);
    neural_t<L> G;
    tnn.init_grad(W, G);
    arma::mat const Data = arma::mat({1.0, 1.0, 2.0, 1.0});
    tnn.error_gradient(W, Data, G);
    tnn.print_layers(G, "Tanh BackProp Gradient: ");
    tnn.numeric_gradient(W, Data, 0.0001, G);
    tnn.print_layers(G, "\nTanh Numeric Gradient: ");

    IdentityNeuralNet<L> inn;
    inn.error_gradient(W, Data, G);
    inn.print_layers(G, "\nIdentity BackProp Gradient: ");
    inn.numeric_gradient(W, Data, 0.0001, G);
    inn.print_layers(G, "\nIdentity Numeric Gradient: ");
}

void q2_a() {
    constexpr std::size_t L = 2;
    IdentityNeuralNet<L> inn;
    dimension_t<L> dims{2, 10, 1};
    arma::mat TrainData;
    arma::mat TestData;
    TrainData.load("../../f_train.txt", arma::file_type::csv_ascii);
    TestData.load("../../f_test.txt", arma::file_type::csv_ascii);
    neural_t<L> const W = inn.create_rand_nn(dims, compute_scale(TrainData, 0.0001));
    // neural_t<L> W = create_dfl_nn<L>(dims, 0.25);
    neural_t<L> learned_W;
    inn.print_layers(W, "W before: ");
    arma::mat Einr;
    inn.variable_rate_gd(W, TrainData, learned_W, Einr, 0.00000001, 1.1, 0.9, 2000000);
    inn.print_layers(learned_W, "W after: ");
    Einr.save("q2_a.txt", arma::file_type::csv_ascii);
    std::cout << "Classification Ein: " << inn.classification_error(learned_W, TrainData) << std::endl 
        << "Classification Eout: " << inn.classification_error(learned_W, TestData) << std::endl;
    arma::mat separator = inn.decision_boundary_2D(learned_W, TestData, 1000);
    std::cout << separator.n_rows << " " << separator.n_cols << std::endl;
    separator.save("q2_a_sep.txt", arma::file_type::csv_ascii);
}

void q2_b() {
    constexpr std::size_t L = 2;
    IdentityNeuralNet<L> inn;
    dimension_t<L> dims{2, 10, 1};
    arma::mat TrainData;
    arma::mat TestData;
    TrainData.load("../../f_train.txt", arma::file_type::csv_ascii);
    TestData.load("../../f_test.txt", arma::file_type::csv_ascii);
    neural_t<L> const W = inn.create_rand_nn(dims, compute_scale(TrainData, 0.0001));
    neural_t<L> learned_W;
    inn.print_layers(W, "W before: ");
    arma::mat Einr;
    inn.variable_rate_reg_gd(W, TrainData, learned_W, Einr, 0.01, 1.1, 0.9, 2000000, 0.01/((double)TrainData.n_rows));
    inn.print_layers(learned_W, "W after: ");
    Einr.save("q2_b.txt", arma::file_type::csv_ascii);
    std::cout << "Classification Ein: " << inn.classification_error(learned_W, TrainData) << std::endl 
        << "Classification Eout: " << inn.classification_error(learned_W, TestData) << std::endl;
    arma::mat separator = inn.decision_boundary_2D(learned_W, TestData, 1000);
    std::cout << separator.n_rows << " " << separator.n_cols << std::endl;
    separator.save("q2_b_sep.txt", arma::file_type::csv_ascii);
}

void q2_c() {
    constexpr std::size_t L = 2;
    IdentityNeuralNet<L> inn;
    dimension_t<L> dims{2, 10, 1};
    arma::mat TrainData;
    arma::mat TestData;
    TrainData.load("../../f_train.txt", arma::file_type::csv_ascii);
    TestData.load("../../f_test.txt", arma::file_type::csv_ascii);
    neural_t<L> const W = inn.create_rand_nn(dims, compute_scale(TrainData, 0.0001));
    neural_t<L> learned_W;
    inn.print_layers(W, "W before: ");
    arma::mat Einr;
    arma::mat Eval;
    inn.variable_rate_early_stop_gd(W, TrainData, learned_W, Einr, Eval, 0.01, 1.1, 0.9, 2000000, 250);
    inn.print_layers(learned_W, "W after: ");
    Einr.save("q2_c_ein.txt", arma::file_type::csv_ascii);
    Eval.save("q2_c_eval.txt", arma::file_type::csv_ascii);
    std::cout << "Classification Ein: " << inn.classification_error(learned_W, TrainData) << std::endl 
        << "Classification Eout: " << inn.classification_error(learned_W, TestData) << std::endl;
    arma::mat separator = inn.decision_boundary_2D(learned_W, TestData, 1000);
    std::cout << separator.n_rows << " " << separator.n_cols << std::endl;
    separator.save("q2_c_sep.txt", arma::file_type::csv_ascii);
}

int main() {
    // q1();
    // tmp();
    // test_fprop_bprop();
    q2_a();
    q2_b();
    q2_c();
    return 0;
}