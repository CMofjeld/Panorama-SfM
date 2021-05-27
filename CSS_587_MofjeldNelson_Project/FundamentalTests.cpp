#include <iostream>
#include <limits>
#include <stdio.h>
#include <iostream>
#include <fstream> 
#include <opencv2/calib3d.hpp>
#include "FundamentalTests.h"
#include "FundamentalSolvers.h"

struct TestResult
{
    // TOOD: If we get time to try other solvers, we can add a property here 
    //       to help indicate which type solver type was used for this test
    long solverTime; // in microseconds
    double normOfError;
};

/// <summary>
/// Evaluate the accuracy of fourPointMethod().
/// </summary>
void testFourPoint() {
   // Generate random points
   const int NUM_POINTS = 4;        // Number of random points to generate
   const float MIN_DEPTH = 6.f;     // Minimum depth from the first camera
   const float MAX_DEPTH = 10.f;    // Maximum depth from the first camera
   const float MAX_X = 20.f;        // Maximum displacement in x-direction
   const float MAX_Y = 20.f;        // Maximum displacement in y-direction

   vector<Point3f> points3d;        // List of randomly generated 3D points
   getRandom3Dpoints(points3d, NUM_POINTS, -MAX_X, MAX_X, -MAX_Y, MAX_Y, MIN_DEPTH, MAX_DEPTH);
   cout << "Random 3D points:" << endl;
   for (auto& point : points3d) cout << point << endl;
   cout << endl;

   // Define arbitrary camera ground truths.
   // Both cameras lie on the unit sphere and have zero skew,
   // unit aspect ratio, and centered principal points.
   // The pose of the first camera is used as the origin of the
   // global coordinate frame.
   const float FOCAL_LEN = 600.f;   // Ground truth focal length
   const float MAX_ROTATION = 10.f * CV_PI / 180.f; // Maximum 10° rotation

   Mat K = Mat::eye(Size(3, 3), CV_32FC1);  // Camera intrinsics
   K.at<float>(0, 0) = FOCAL_LEN;
   K.at<float>(1, 1) = FOCAL_LEN;

   Mat R; // Rotation matrix
   getRandom3DRotationMat(R, MAX_ROTATION, MAX_ROTATION, MAX_ROTATION);
   Mat rvec;   // Rotation vector
   Rodrigues(R, rvec);
   Mat t; // Translation vector
   R.col(2).copyTo(t);
   t.at<float>(2, 0) -= 1;

   // Project points to both cameras
   vector<Point2f> projectedPoints1, projectedPoints2;
   Mat zeroVec = Mat::zeros(3, 1, DataType<float>::type);
   projectPoints(points3d, zeroVec, zeroVec, K, noArray(), projectedPoints1);
   projectPoints(points3d, rvec, t, K, noArray(), projectedPoints2);
   cout << "Projected points 1:" << endl;
   for (auto& point : projectedPoints1) cout << point << endl;
   cout << endl;
   cout << "Projected points 2:" << endl;
   for (auto& point : projectedPoints2) cout << point << endl;
   cout << endl;

   // Get four point estimations
   vector<Mat> solutions;
   Mat homogeneousP1, homogeneousP2;
   hconcat(Mat(projectedPoints1.size(), 2, CV_32F, projectedPoints1.data()), Mat::ones(projectedPoints1.size(), 1, CV_32F), homogeneousP1);
   hconcat(Mat(projectedPoints2.size(), 2, CV_32F, projectedPoints2.data()), Mat::ones(projectedPoints2.size(), 1, CV_32F), homogeneousP2);
   fourPointMethod(homogeneousP1, homogeneousP2, solutions);

   // Calculate ground truth fundamental matrix
   Mat F, bestEstimate;
   fundamentalFromKRT(F, K, K, R, t);
   F = F / norm(F);
   cout << "Ground truth Fundamental:" << endl;
   cout << F << endl << endl;

   // Find the estimated solution with the lowest error
   double bestError = DBL_MAX;
   for (auto& solution : solutions) {
      double curError = norm(F - solution);
      if (curError < bestError) {
         bestEstimate = solution;
         bestError = curError;
      }
   }
   cout << "Estimated Fundamental:" << endl;
   cout << bestEstimate << endl << endl;
   cout << "Frobenius norm of error:" << endl;
   cout << bestError << endl << endl;
}

/// <summary>
/// Evaluate the performance of estimateFundamentalMatrix().
/// </summary>
void testEstimateFundamentalMatrix() {
   // Generate random points
   const int NUM_POINTS = 50;     // Number of random points to generate
   const float MIN_DEPTH = 6.f;     // Minimum depth from the first camera
   const float MAX_DEPTH = 10.f;    // Maximum depth from the first camera
   const float MAX_X = 20.f;        // Maximum displacement in x-direction
   const float MAX_Y = 20.f;        // Maximum displacement in y-direction

   vector<Point3f> points3d;        // List of randomly generated 3D points
   getRandom3Dpoints(points3d, NUM_POINTS, -MAX_X, MAX_X, -MAX_Y, MAX_Y, MIN_DEPTH, MAX_DEPTH);

   // Define arbitrary camera ground truths.
   // Both cameras lie on the unit sphere and have zero skew,
   // unit aspect ratio, and centered principal points.
   // The pose of the first camera is used as the origin of the
   // global coordinate frame.
   const float FOCAL_LEN = 600.f;   // Ground truth focal length
   const float MAX_ROTATION = 10.f * CV_PI / 180.f; // Maximum 10° rotation

   Mat K = Mat::eye(Size(3, 3), CV_32FC1);  // Camera intrinsics
   K.at<float>(0, 0) = FOCAL_LEN;
   K.at<float>(1, 1) = FOCAL_LEN;

   Mat R; // Rotation matrix
   getRandom3DRotationMat(R, MAX_ROTATION, MAX_ROTATION, MAX_ROTATION);
   Mat rvec;   // Rotation vector
   Rodrigues(R, rvec);
   Mat t; // Translation vector
   R.col(2).copyTo(t);
   t.at<float>(2, 0) -= 1;

   // Project points to both cameras
   vector<Point2f> projectedPoints1, projectedPoints2;
   Mat zeroVec = Mat::zeros(3, 1, DataType<float>::type);
   projectPoints(points3d, zeroVec, zeroVec, K, noArray(), projectedPoints1);
   projectPoints(points3d, rvec, t, K, noArray(), projectedPoints2);

   // Get four point estimations
   Mat homogeneousP1, homogeneousP2;
   hconcat(Mat(projectedPoints1.size(), 2, CV_32F, projectedPoints1.data()), Mat::ones(projectedPoints1.size(), 1, CV_32F), homogeneousP1);
   hconcat(Mat(projectedPoints2.size(), 2, CV_32F, projectedPoints2.data()), Mat::ones(projectedPoints2.size(), 1, CV_32F), homogeneousP2);
   Mat bestEstimate = estimateFundamentalMatrix(homogeneousP1, homogeneousP2);

   // Calculate ground truth fundamental matrix
   Mat F;
   fundamentalFromKRT(F, K, K, R, t);
   F = F / norm(F);
   cout << "Ground truth Fundamental:" << endl;
   cout << F << endl << endl;

   // Find the estimated solution with the lowest error
   cout << "Estimated Fundamental:" << endl;
   cout << bestEstimate << endl << endl;
   cout << "Frobenius norm of error:" << endl;
   cout << norm(F - bestEstimate) << endl << endl;
}

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
void getRandom3Dpoints(vector<Point3f>& points3d, int numPoints, float xMin, float xMax, float yMin, float yMax, float zMin, float zMax) {
   static RNG rng(12345);

   for (int i = 0; i < numPoints; i++)
   {
      Point3f randomPoint;
      randomPoint.x = rng.uniform(xMin, xMax);
      randomPoint.y = rng.uniform(yMin, yMax);
      randomPoint.z = rng.uniform(zMin, zMax);
      points3d.push_back(randomPoint);
   }
}

/// <summary>
/// Generate a set of random 3D points from a uniform distribution.
/// </summary>
/// <param name="points3d">[Output] Set of random 3D points</param>
/// <param name="numPoints">Number of points to generate
void getRandom3Dpoints(vector<Point3f>& points3d, int numPoints) {
    const float MIN_DEPTH = 6.f;     // Minimum depth from the first camera
    const float MAX_DEPTH = 10.f;    // Maximum depth from the first camera
    const float MAX_X = 20.f;        // Maximum displacement in x-direction
    const float MAX_Y = 20.f;        // Maximum displacement in y-direction
    static RNG rng(12345);

    for (int i = 0; i < numPoints; i++)
    {
        Point3f randomPoint;
        randomPoint.x = rng.uniform(-MAX_X, MAX_X);
        randomPoint.y = rng.uniform(-MAX_Y, MAX_Y);
        randomPoint.z = rng.uniform(MIN_DEPTH, MAX_DEPTH);
        points3d.push_back(randomPoint);
    }
}

/// <summary>
/// Generate 3D rotation matrix with random rotations selected from a uniform distribution.
/// </summary>
/// <param name="rotationMat">[Output] Rotation matrix</param>
/// <param name="xRotMax">Maximum magnitude of rotation around x-axis (radians)</param>
/// <param name="yRotMax">Maximum magnitude of rotation around y-axis (radians)</param>
/// <param name="zRotMax">Maximum magnitude of rotation around z-axis (radians)</param>
void getRandom3DRotationMat(Mat& rotationMat, float xRotMax, float yRotMax, float zRotMax) {
   static RNG rng(12345);
   float x_angle = rng.uniform(-xRotMax, xRotMax);
   float y_angle = rng.uniform(-yRotMax, yRotMax);
   float z_angle = rng.uniform(-zRotMax, zRotMax);
   Mat R_x = (Mat_<float>(3, 3) <<
      1, 0, 0,
      0, cos(x_angle), -sin(x_angle),
      0, sin(x_angle), cos(x_angle));
   Mat R_y = (Mat_<float>(3, 3) <<
      cos(y_angle), 0, sin(y_angle),
      0, 1, 0,
      -sin(y_angle), 0, cos(y_angle));
   Mat R_z = (Mat_<float>(3, 3) <<
      cos(z_angle), -sin(z_angle), 0,
      sin(z_angle), cos(z_angle), 0,
      0, 0, 1);
   rotationMat = R_z * R_y * R_x;
}

/// <summary>
/// Generate 3D rotation matrix with random rotations selected from a uniform distribution.
/// </summary>
/// <param name="rotationMat">[Output] Rotation matrix</param>
void getRandom3DRotationMat(Mat& rotationMat) {
    static RNG rng(12345);
    const float MAX_ROTATION = 10.f * CV_PI / 180.f; // Maximum 10° rotation
    float x_angle = rng.uniform(-MAX_ROTATION, MAX_ROTATION);
    float y_angle = rng.uniform(-MAX_ROTATION, MAX_ROTATION);
    float z_angle = rng.uniform(-MAX_ROTATION, MAX_ROTATION);
    Mat R_x = (Mat_<float>(3, 3) <<
        1, 0, 0,
        0, cos(x_angle), -sin(x_angle),
        0, sin(x_angle), cos(x_angle));
    Mat R_y = (Mat_<float>(3, 3) <<
        cos(y_angle), 0, sin(y_angle),
        0, 1, 0,
        -sin(y_angle), 0, cos(y_angle));
    Mat R_z = (Mat_<float>(3, 3) <<
        cos(z_angle), -sin(z_angle), 0,
        sin(z_angle), cos(z_angle), 0,
        0, 0, 1);
    rotationMat = R_z * R_y * R_x;
}

/// <summary>
/// Get Fundamental matrix from known camera intrinsics, rotation, and translation.
/// </summary>
/// <param name="rotationMat">[Output] Fundamental matrix</param>
/// <param name="K1">First camera's intrinsic matrix</param>
/// <param name="K2">Second camera's intrinsic matrix</param>
/// <param name="R">Rotation matrix</param>
/// <param name="t">Translation vector</param>
void fundamentalFromKRT(Mat& F, const Mat& K1, const Mat& K2, const Mat& R, const Mat& t) {
   // Reference for this code: https://sourishghosh.com/2016/fundamental-matrix-from-camera-matrices/
   Mat A = K1 * R.t() * t;
   Mat C = (Mat_<float>(3, 3) <<
      0, -A.at<float>(2, 0), A.at<float>(1, 0),
      A.at<float>(2, 0), 0, -A.at<float>(0, 0),
      -A.at<float>(1, 0), A.at<float>(0, 0), 0);
   F = (K2.inv()).t() * R * K1.t() * C;
}

/// <summary>
/// Attempt to save results to a file
/// </summary>
/// <param name="fileName">Desired name of the file</param>
/// <param name="results">Results to log</param>
void saveTrailResults(String fileName, vector<TestResult> results)
{
    struct stat fileInfo;

    // Delete the previously existing file (if one exists)
    if (stat(fileName.c_str(), &fileInfo) == 0)
    {
        if (remove(fileName.c_str()) != 0)
        {
            printf("Error occurred deleting existing file");
            return;
        }
    }

    // Create the file, and log the results to the file
    ofstream resultFile(fileName.c_str());
    resultFile.precision(17);
    for (int i = 0; i < results.size(); i++)
    {
        resultFile << i << "," << results.at(i).normOfError << "," << results.at(i).solverTime << endl;
    }

    resultFile.close();
}

/// <summary>
/// Run the zero-noise trials
/// </summary>
/// <param name="numOfTrials">Desired number of trails to run</param>
void runZeroNoiseTrials(int numOfTrials)
{
    const int NUM_OF_RANDOM_POINTS = 1000;
    const int MIN_NUMBER_OF_RANDOM_ROTATIONS = 400;
    const int MAX_NUMBER_OF_RANDOM_ROTATIONS = 2000;

    const float FOCAL_LEN = 600.f;   // Ground truth focal length
    Mat K = Mat::eye(Size(3, 3), CV_32FC1);  // Camera intrinsics
    
    vector<double> errorNorms;
    vector<TestResult> results;

    K.at<float>(0, 0) = FOCAL_LEN;
    K.at<float>(1, 1) = FOCAL_LEN;
    
    for (int i = 0; i < numOfTrials; i++)
    {
        RNG rng(i);
        
        double normOfError;
        TestResult result;
        
        // First generate all of our 3D points
        vector<Point3f> points3d;
        getRandom3Dpoints(points3d, NUM_OF_RANDOM_POINTS);
        
        // Generate the random rotation
        Mat rotationMat;
        getRandom3DRotationMat(rotationMat);

        // Convert rotation and translation vectors
        Mat rotationVector, translationVector;
        Rodrigues(rotationMat, rotationVector);
        rotationMat.col(2).copyTo(translationVector);
        translationVector.at<float>(2, 0) -= 1;

        // Project the points into each camera rotation
        Mat zeroVec = Mat::zeros(3, 1, DataType<float>::type);
        vector<Point2f> projectedPoints1, projectedPoints2;
        projectPoints(points3d, zeroVec, zeroVec, K, noArray(), projectedPoints1); // TODO: Should we consider generating a random rotation for this too?
        projectPoints(points3d, rotationVector, translationVector, K, noArray(), projectedPoints2);

        // Get four point estimations
        Mat homogeneousP1, homogeneousP2;
        hconcat(Mat(projectedPoints1.size(), 2, CV_32F, projectedPoints1.data()), Mat::ones(projectedPoints1.size(), 1, CV_32F), homogeneousP1);
        hconcat(Mat(projectedPoints2.size(), 2, CV_32F, projectedPoints2.data()), Mat::ones(projectedPoints2.size(), 1, CV_32F), homogeneousP2);

        auto start = high_resolution_clock::now();
        Mat bestEstimate = estimateFundamentalMatrix(homogeneousP1, homogeneousP2);
        auto stop = high_resolution_clock::now();

        // Calculate ground truth fundamental matrix
        Mat groundTruthFundamental;
        fundamentalFromKRT(groundTruthFundamental, K, K, rotationMat, translationVector);
        groundTruthFundamental = groundTruthFundamental / norm(groundTruthFundamental);

        // Find the estimated solution with the lowest error
        auto duration = duration_cast<microseconds>(stop - start);
        result.normOfError = norm(groundTruthFundamental - bestEstimate);
        result.solverTime = duration.count();

        if (result.normOfError > 1)
        {
            //cout << "Ground truth Fundamental:" << endl;
            //cout << groundTruthFundamental << endl << endl;

            //// Find the estimated solution with the lowest error
            //cout << "Estimated Fundamental:" << endl;
            //cout << bestEstimate << endl << endl;

            Mat absGroundTruth = abs(groundTruthFundamental);
            Mat absCalculated = abs(bestEstimate);
            result.normOfError = norm(absGroundTruth - absCalculated);
        }

        results.push_back(result);
    }

    saveTrailResults("zero_noise_trail.csv", results);
}