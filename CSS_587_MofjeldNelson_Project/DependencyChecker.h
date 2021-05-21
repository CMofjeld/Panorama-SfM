/*
* CSS 587 - Advance Computer Vision
* Spring 2021
* Carl Mofjeld, Drew Nelson
* 
* Description:
* Header file for the dependency checker. The dependency check simply runs a few 
* methods from some of the dependencies required by OpenCV's SfM and Viz modules.
* Most of the methods in this class are short operations usually found in the
* "Hello, world!" applications for each dependency.
*/

#pragma once

#include <iostream>
#include <glog/logging.h>
#include <Eigen/Dense>
#include "gflags/gflags.h"
#include <math.h>
#include "ceres/ceres.h"

using namespace std;
using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solve;
using ceres::Solver;

// A templated cost functor that implements the residual r = 10 -
// x. The method operator() is templated so that we can then use an
// automatic differentiation wrapper around it to generate its
// derivatives.
struct CostFunctor {
    template <typename T>
    bool operator()(const T* const x, T* residual) const {
        residual[0] = 10.0 - x[0];
        return true;
    }
};

/// <summary>
/// A quick method to run a method from glog to confirm we're able to
/// run methods from this dependency
/// </summary>
void testGlog();

/// <summary>
/// A quick method to run a method from eigen to confirm we're able to
/// run methods from this dependency
/// </summary>
void testEigen();

/// <summary>
/// A quick method to run a method from gflags to confirm we're able to
/// run methods from this dependency
/// </summary>
void testGFlags();

/// <summary>
/// A quick method to run a method from ceres to ensure we're able to
/// run methods from this dependency
/// </summary>
void testCeres();

/// <summary>
/// Run methods from the GLog, GFlags, Eigen, and Ceres Solver
/// dependencies to ensure they've loaded and can run correctly.
/// </summary>
/// <returns>True if all dependencies ran without error. False otherwise</returns>
bool testDependencies();