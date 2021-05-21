/*
* CSS 587 - Advance Computer Vision
* Spring 2021
* Carl Mofjeld, Drew Nelson
*
* Description:
* Implementations for the the dependency checker. The dependency check 
* simply runs a few methods from some of the dependencies required by 
* OpenCV's SfM and Viz modules. Most of the methods in this class are 
* short operations usually found in the "Hello, world!" applications 
* for each dependency.
*/

#include "DependencyChecker.h"

/// <summary>
/// A quick method to run a method from glog to confirm we're able to
/// run methods from this dependency
/// </summary>
void testGlog()
{
    // Simple test to ensure we have glog configured correctly
    google::InitGoogleLogging("");
    LOG(INFO) << "GLOG configured correctly";
    printf("Glog configured correctly\n");
}

/// <summary>
/// A quick method to run a method from eigen to confirm we're able to
/// run methods from this dependency
/// </summary>
void testEigen()
{
    Eigen::MatrixXd m(2, 2);
    m(0, 0) = 3;
    m(1, 0) = 2.5;
    m(0, 1) = -1;
    m(1, 1) = m(1, 0) + m(0, 1);
    printf("Eigen configured correctly\n");
}

/// <summary>
/// A quick method to run a method from gflags to confirm we're able to
/// run methods from this dependency
/// </summary>
void testGFlags()
{
    gflags::SetVersionString("1.0.0");
    printf("GFlags configured correctly\n");
}

/// <summary>
/// A quick method to run a method from ceres to ensure we're able to
/// run methods from this dependency
/// </summary>
void testCeres()
{
    // The variable to solve for with its initial value. It will be
    // mutated in place by the solver.
    double x = 0.5;
    const double initial_x = x;
    // Build the problem.
    Problem problem;
    // Set up the only cost function (also known as residual). This uses
    // auto-differentiation to obtain the derivative (jacobian).
    CostFunction* cost_function =
        new AutoDiffCostFunction<CostFunctor, 1, 1>(new CostFunctor);
    problem.AddResidualBlock(cost_function, nullptr, &x);
    // Run the solver!
    Solver::Options options;
    options.minimizer_progress_to_stdout = true;
    Solver::Summary summary;
    Solve(options, &problem, &summary);

    printf("Ceres configured correctly\n");
}

/// <summary>
/// Run methods from the GLog, GFlags, Eigen, and Ceres Solver
/// dependencies to ensure they've loaded and can run correctly.
/// </summary>
/// <returns>True if all dependencies ran without error. False otherwise</returns>
bool testDependencies()
{
    bool dependenciesLoadedCorrectly = true;
    try
    {
        testGlog();
        testEigen();
        testGFlags();
        testCeres();
    }
    catch (...)
    {
        printf("Error occurred in dependency configuration\n");
        dependenciesLoadedCorrectly = false;
    }
    return dependenciesLoadedCorrectly;
}