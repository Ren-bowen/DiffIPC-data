// Link against the patched legacy PolySolve; no simulation or mesh required.
#include <polysolve/nonlinear/Solver.hpp>
#include <polysolve/nonlinear/Problem.hpp>
#include <spdlog/spdlog.h>
#include <iostream>
#include <stdexcept>

class Quadratic final : public polysolve::nonlinear::Problem
{
public:
    double value(const TVector& x) override { return 0.5 * x.squaredNorm(); }
    void gradient(const TVector& x, TVector& g) override { g = x; }
    void hessian(const TVector& x, THessian& h) override
    {
        h.resize(x.size(), x.size());
        h.setIdentity();
    }
};

int main()
{
    using polysolve::nonlinear::Solver;
    using polysolve::json;
    Quadratic problem;
    json params = {{"solver", "Newton"}, {"grad_norm", 0.0},
                   {"first_grad_norm_tol", 0.0}, {"x_delta", 1.0},
                   {"x_delta_linf_absolute_tol", 1.0}, {"max_iterations", 10}};
    json linear = {{"solver", "Eigen::SimplicialLDLT"}};
    auto make_solver = [&](double scale) {
        return Solver::create(params, linear, scale, *spdlog::default_logger());
    };
    // Linf=0.75 < 1, while L2>1: must stop before the first line search.
    auto solver = make_solver(1000);
    if (solver->stop_criteria().xDelta != 1.0)
        throw std::runtime_error("Absolute tolerance was scaled");
    Eigen::VectorXd x = Eigen::VectorXd::Constant(3, 0.75);
    solver->minimize(problem, x);
    if ((x.array() != 0.75).any())
        throw std::runtime_error("Did not stop on the pre-line-search Linf direction");
    // With the same large characteristic length, direction 2 must not stop.
    solver = make_solver(1000);
    x = Eigen::VectorXd::Constant(3, 2.0);
    solver->minimize(problem, x);
    if (x.norm() > 1e-12)
        throw std::runtime_error("Stopped above the absolute threshold");
    // The comparison is strict, as in Unified.
    solver = make_solver(1000);
    x = Eigen::VectorXd::Constant(3, 1.0);
    solver->minimize(problem, x);
    if (x.norm() > 1e-12)
        throw std::runtime_error("Stopped at equality instead of strict less-than");
    // Opting out retains legacy L2 stopping.
    params["x_delta_linf_absolute_tol"] = 0.0;
    solver = make_solver(1);
    x = Eigen::VectorXd::Constant(3, 0.75);
    solver->minimize(problem, x);
    if (x.norm() > 1e-12)
        throw std::runtime_error("Legacy L2 behavior changed");
    std::cout << "PASS: absolute scaling, Linf vs L2, pre-line-search stop, strict inequality, legacy default\n";
}
