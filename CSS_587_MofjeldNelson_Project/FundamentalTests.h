/*
* CSS 587 - Advance Computer Vision
* Spring 2021
* Carl Mofjeld, Drew Nelson
*
* Description:
* Contains the methods used to help test the fundamental solvers. The methods here
* attempt to follow the methods trials setup as described in the paper
*/

#pragma once
#include <opencv2/core.hpp>
#include <vector>
#include "FileHelper.h"
#include "FundamentalSolvers.h"

using namespace cv;
using namespace std;
using namespace std::chrono;

/// <summary>
/// Utility structure to help encapsulate test information
/// </summary>
struct TestResult
{
    SolverType solverType;
    long solverTime; // in microseconds
    double normOfError;
    int iterationIndex;
};

/// <summary>
/// Evaluate the performance of estimateFundamentalMatrix().
/// </summary>
void SfmTesting(Mat& fundamentalMatrix, vector<Point2f>& projectedPoints1, vector<Point2f>& projectedPoints2);

/// <summary>
/// Generate a set of random 3D points from a uniform distribution.
/// </summary>
/// <param name="points3d">[Output] Set of random 3D points</param>
/// <param name="xMin">Minimum x-coordinate</param>
/// <param name="xMax">Maximum x-coordinate</param>
/// <param name="yMin">Minimum y-coordinate</param>
/// <param name="yMax">Maximum y-coordinate</param>
/// <param name="zMin">Minimum z-coordinate</param>
/// <param name="zMax">Maximum z-coordinate</param>
void getRandom3Dpoints(vector<Point3f>& points3d, int numPoints, float xMin, float xMax, float yMin, float yMax, float zMin, float zMax);

/// <summary>
/// Generate 3D rotation matrix with random rotations selected from a uniform distribution.
/// </summary>
/// <param name="rotationMat">[Output] Rotation matrix</param>
/// <param name="xRotMax">Maximum magnitude of rotation around x-axis (radians)</param>
/// <param name="yRotMax">Maximum magnitude of rotation around y-axis (radians)</param>
/// <param name="zRotMax">Maximum magnitude of rotation around z-axis (radians)</param>
void getRandom3DRotationMat(Mat& rotationMat, float xRotMax, float yRotMax, float zRotMax);

/// <summary>
/// Get Fundamental matrix from known camera intrinsics, rotation, and translation.
/// </summary>
/// <param name="rotationMat">[Output] Fundamental matrix</param>
/// <param name="K1">First camera's intrinsic matrix</param>
/// <param name="K2">Second camera's intrinsic matrix</param>
/// <param name="R">Rotation matrix</param>
/// <param name="t">Translation vector</param>
void fundamentalFromKRT(Mat& F, const Mat& K1, const Mat& K2, const Mat& R, const Mat& t);

/// <summary>
/// Run the zero-noise trials
/// </summary>
/// <param name="numOfTrials">Desired number of trails to run</param>
void runZeroNoiseTrials(int numOfTrials);

/// <summary>
/// Run the trials that apply guassian noise
/// </summary>
/// <param name="numOfTrialsPerLevel">Number of trails to run per gaussian level</param>
void runNoiseTrials(int numOfTrialsPerLevel);

/// <summary>
/// Run both the zero-noise and noisy trials. Ensures the result directory is created before running the trials
/// </summary>
/// <param name="numOfZeroNoiseTrails">Number of zero-noise trails to run.</param>
/// <param name="numOfTrailsPerNoiseLevel">Number of trials to run per noise level</param>
void runAllTrials(int numOfZeroNoiseTrails, int numOfTrailsPerNoiseLevel);