#include <iostream>
#include <limits>
#include <stdio.h>
#include <iostream>
#include <fstream> 
#include <random>
#include <opencv2/calib3d.hpp>
#include "FundamentalTests.h"
#include "FundamentalSolvers.h"

// #define LOG_DETAILS

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
/// Utility method to create a print friendly representation of the solver type
/// </summary>
/// <param name="solverType">Solver type to convert</param>
/// <returns>Solver type as string</returns>
string solverTypeToString(SolverType solverType)
{
    switch (solverType)
    {
        case FourPoint:
            return "FourPoint";
        case SixPoint:
            return "SixPoint";
        case CV_SevenPoint:
            return "CV_SevenPoint";
        case CV_EightPoint:
            return "CV_EightPoint";
    }
    return "Unknown";
}

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
   const float MAX_ROTATION = 10.f * CV_PI / 180.f; // Maximum 10� rotation

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
   const float MAX_ROTATION = 10.f * CV_PI / 180.f; // Maximum 10� rotation

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
/// <param name="numPoints">Number of points to generate
/// <returns>List of generated points</returns>
vector<Point3f> getRandom3Dpoints(int numPoints) {
    const float MIN_DEPTH = 6.f;     // Minimum depth from the first camera
    const float MAX_DEPTH = 10.f;    // Maximum depth from the first camera
    const float MAX_X = 20.f;        // Maximum displacement in x-direction
    const float MAX_Y = 20.f;        // Maximum displacement in y-direction
    static RNG rng(12345);
    vector<Point3f> points3d;

    for (int i = 0; i < numPoints; i++)
    {
        Point3f randomPoint;
        randomPoint.x = rng.uniform(-MAX_X, MAX_X);
        randomPoint.y = rng.uniform(-MAX_Y, MAX_Y);
        randomPoint.z = rng.uniform(MIN_DEPTH, MAX_DEPTH);
        points3d.push_back(randomPoint);
    }
    return points3d;
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
    const float MAX_ROTATION = 10.f * CV_PI / 180.f; // Maximum 10� rotation
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
    if (!deleteFile(fileName))
    {
        return;
    }

    // Create the file, and log the results to the file
    ofstream resultFile(fileName.c_str());
    resultFile.precision(17);

#ifdef LOG_DETAILS
    cout << "Results for " << fileName << ":" << endl;
#endif
    for (int i = 0; i < results.size(); i++)
    {
        resultFile << results.at(i).iterationIndex << "," << solverTypeToString(results.at(i).solverType) << "," << results.at(i).normOfError << "," << results.at(i).solverTime << endl;

#ifdef LOG_DETAILS
        cout << results.at(i).iterationIndex << "," << solverTypeToString(results.at(i).solverType) << "," << results.at(i).normOfError << "," << results.at(i).solverTime << endl;
#endif
    }

    resultFile.close();
}

/// <summary>
/// Temporary method to handle the strange behavior we're seeing when our fundamental
/// matrix has a high norm of error. The beahvior we're seeing is that the values between
/// the caulculated and ground truth matrices are similar with the exception that their sign is the opposed
/// which is yielding the high error. 
/// 
/// To deal with this temporarily, we'll flip the sign in our calculated matrix and recalculate the error
/// </summary>
/// <param name="groundTruthFundamental">Ground truth fundamental matrix</param>
/// <param name="calculatedFundamental">Best estimated calculated fundamental matrix</param>
/// <returns></returns>
double handleHighErrorResult(Mat groundTruthFundamental, Mat calculatedFundamental)
{
#ifdef LOG_DETAILS
    cout << "High error found with calculated fundamental matrix. Applying temporary fix" << endl;
    cout << "Ground truth Fundamental:" << endl;
    cout << groundTruthFundamental << endl << endl;

    cout << "Estimated Fundamental:" << endl;
    cout << calculatedFundamental << endl << endl;
#endif

    Mat flippedResult = calculatedFundamental * -1;
    return norm(groundTruthFundamental - flippedResult);
}

/// <summary>
/// Calculate the fundamental matrix and time how long it takes. 
/// </summary>
/// <param name="homogeneousP1">Homogeneous matrix from project points of one camera</param>
/// <param name="homogeneousP2">Homogeneous matrix from project points of the second camera</param>
/// <param name="bestEstimate">[Output] Best esetimated fundamental matrix</param>
/// <returns>Number of microseconds it took to determine the best estimated fundamental matrix</returns>
long calculateAndTimeFundamentalMatrix(CustomSolver solver, Mat homogeneousP1, Mat homogeneousP2, Mat& bestEstimate)
{
    auto start = high_resolution_clock::now();
    bestEstimate = estimateFundamentalMatrix(solver, homogeneousP1, homogeneousP2);
    auto stop = high_resolution_clock::now();

    // Find the estimated solution with the lowest error
    auto duration = duration_cast<microseconds>(stop - start);
    return duration.count();
}

/// <summary>
/// Get a rotation and translation vector from a rotation matrix
/// </summary>
/// <param name="rotationMat"></param>
/// <param name="rotationVector">[Output] Resulting rotation vector</param>
/// <param name="translationVector">[Output] Resulting translation vector</param>
void getRotationTranslationVectorsFromRotationMat(Mat rotationMat, Mat &rotationVector, Mat &translationVector)
{
    Rodrigues(rotationMat, rotationVector);
    rotationMat.col(2).copyTo(translationVector);
    translationVector.at<float>(2, 0) -= 1;
}

/// <summary>
/// Compute the norm of error between a ground truth fundamental matrix and a calculated fundamental matrix
/// </summary>
/// <param name="groundTruthFundamental">Ground truth fudnamental matrix</param>
/// <param name="bestEstimate">Calculated fundamental matrix</param>
/// <returns>Norm of error</returns>
double computeNormOfError(Mat groundTruthFundamental, Mat bestEstimate)
{
    double normOfError = norm(groundTruthFundamental - bestEstimate);
    if (normOfError > 1)
    {
        normOfError = handleHighErrorResult(groundTruthFundamental, bestEstimate);
    }
    return normOfError;
}

/// <summary>
/// Duplicate a list of points and add noiose to the points. If the noise level <= 0, then a cloned list
/// is simply returned
/// </summary>
/// <param name="originalPoints">Points to add noise to</param>
/// <param name="noiseLevel">Noise level to add to the points. Noise level </param>
/// <returns></returns>
vector<Point3f> addNoiseToPoints(vector<Point3f> originalPoints, float noiseLevel)
{
    vector<Point3f> noisyPoints(originalPoints);
    if (noiseLevel > 0)
    {
        default_random_engine gen;
        normal_distribution<float> dist(0.0f, noiseLevel);

        for (int i = 0; i < noisyPoints.size(); i++)
        {
            noisyPoints[i].x = noisyPoints[i].x + dist(gen);
            noisyPoints[i].y = noisyPoints[i].y + dist(gen);
            noisyPoints[i].z = noisyPoints[i].z + dist(gen);
        }
    }
    return noisyPoints;
}

/// <summary>
/// Get the normalized ground truth fundamental matrix 
/// </summary>
/// <param name="intrinsicMatrix">Camera's intrinsic matrix</param>
/// <param name="rotationMat">Camera rotations</param>
/// <param name="translationVector">Camera translations</param>
/// <returns></returns>
Mat getNormalizedGroundTruthMatrix(Mat intrinsicMatrix, Mat rotationMat, Mat translationVector)
{
    Mat groundTruthFundamental;
    fundamentalFromKRT(groundTruthFundamental, intrinsicMatrix, intrinsicMatrix, rotationMat, translationVector);
    groundTruthFundamental = groundTruthFundamental / norm(groundTruthFundamental);
    return groundTruthFundamental;
}

/// <summary>
/// Get the homogenous matricies by projecting the 3D points into each camera space. 
/// </summary>
/// <param name="points3d1">Points to project to the first camera</param>
/// <param name="points3d2">Points to project to the second camera</param>
/// <param name="intrinsicMatrix">Camera intrinsic matrix</param>
/// <param name="rotationVector">Camera's rotation movement (for the second camera)</param>
/// <param name="translationVector">Camera's translation movement (for the second camera)</param>
/// <param name="homogeneousP1">[Output] Homogenous mat for the first points</param>
/// <param name="homogeneousP2">[Output] Homogenous mat for the second points</param>
void getHomogeneousMatsFrom3DPoints(vector<Point3f> points3d1, vector<Point3f> points3d2, Mat intrinsicMatrix, Mat rotationVector, Mat translationVector, Mat &homogeneousP1, Mat &homogeneousP2)
{
    // Project the points into each camera rotation
    Mat zeroVec = Mat::zeros(3, 1, DataType<float>::type);
    vector<Point2f> projectedPoints1, projectedPoints2;
    projectPoints(points3d1, zeroVec, zeroVec, intrinsicMatrix, noArray(), projectedPoints1); // TODO: Should we consider generating a random rotation for this too?
    projectPoints(points3d2, rotationVector, translationVector, intrinsicMatrix, noArray(), projectedPoints2);

    hconcat(Mat(projectedPoints1.size(), 2, CV_32F, projectedPoints1.data()), Mat::ones(projectedPoints1.size(), 1, CV_32F), homogeneousP1);
    hconcat(Mat(projectedPoints2.size(), 2, CV_32F, projectedPoints2.data()), Mat::ones(projectedPoints2.size(), 1, CV_32F), homogeneousP2);
}

/// <summary>
/// Get the homogenous matricies by projecting the 3D points into each camera space. 
/// </summary>
/// <param name="points3d1">Points to project to both cameras</param>
/// <param name="intrinsicMatrix">Camera intrinsic matrix</param>
/// <param name="rotationVector">Camera's rotation movement (for the second camera)</param>
/// <param name="translationVector">Camera's translation movement (for the second camera)</param>
/// <param name="homogeneousP1">[Output] Homogenous mat for the first points</param>
/// <param name="homogeneousP2">[Output] Homogenous mat for the second points</param>
void getHomogeneousMatsFrom3DPoints(vector<Point3f> points3d, Mat intrinsicMatrix, Mat rotationVector, Mat translationVector, Mat& homogeneousP1, Mat& homogeneousP2)
{
    getHomogeneousMatsFrom3DPoints(points3d, points3d, intrinsicMatrix, rotationVector, translationVector, homogeneousP1, homogeneousP2);
}

/// <summary>
/// Create an intrinsic matrix with the focal point in the diagonal, except the bottom right corner which should be 1
/// </summary>
/// <param name="focalLength">Camera focal length to populate the intrinsic matrix</param>
/// <returns>Intrinsic matrix</returns>
Mat createIntrinsicMatrix(float focalLength)
{
    Mat intrinsicMatrix = Mat::eye(Size(3, 3), CV_32FC1);
    intrinsicMatrix.at<float>(0, 0) = focalLength;
    intrinsicMatrix.at<float>(1, 1) = focalLength;
    intrinsicMatrix.at<float>(2, 2) = 1;
    return intrinsicMatrix;
}

/// <summary>
/// Estimate the fundamental matrix using each available solver
/// </summary>
/// <param name="iterationIndex">Trial index</param>
/// <param name="homogeneousP1">Homogeneous points from first matrix</param>
/// <param name="homogeneousP2">Homogeneous points from second matrix</param>
/// <param name="intrinsicMatrix">Camera intrinsic matrix</param>
/// <param name="rotationMat">Camera rotation matrix</param>
/// <param name="translationVector">Camera translation vector</param>
/// <returns></returns>
vector<TestResult> estimateFundamentals(int iterationIndex, Mat homogeneousP1, Mat homogeneousP2, Mat intrinsicMatrix, Mat rotationMat, Mat translationVector)
{
    vector<TestResult> results;

    //TODO: If we have time, we should consider moving this list elsewhere since it's being redefined for each trail attempt
    CustomSolver solvers[] = { FourPointSolver, /*SixPointSolver, */SevenPointSolver, EightPointSolver }; 
    Mat groundTruthFundamental = getNormalizedGroundTruthMatrix(intrinsicMatrix, rotationMat, translationVector);
    for (auto solver : solvers)
    {
        TestResult result;
        Mat bestEstimate;
        double timeInMicroseconds = calculateAndTimeFundamentalMatrix(solver, homogeneousP1, homogeneousP2, bestEstimate);

        result.iterationIndex = iterationIndex;
        result.normOfError = computeNormOfError(groundTruthFundamental, bestEstimate);
        result.solverTime = timeInMicroseconds;
        result.solverType = solver.solverType;
        results.push_back(result);
    }
    return results;
}

/// <summary>
/// Run both the zero-noise and noisy trials. Ensures the result directory is created before running the trials
/// </summary>
/// <param name="numOfZeroNoiseTrails">Number of zero-noise trails to run.</param>
/// <param name="numOfTrailsPerNoiseLevel">Number of trials to run per noise level</param>
void runAllTrials(int numOfZeroNoiseTrails, int numOfTrailsPerNoiseLevel)
{
    const std::string resultDir = "results";
    if (!createDirectory(resultDir))
    {
        printf("Canceling trail tests - failed to create result directory");
        return;
    }

    runZeroNoiseTrials(numOfZeroNoiseTrails);
    runNoiseTrials(numOfTrailsPerNoiseLevel);
}

/// <summary>
/// Run the trials that apply guassian noise
/// </summary>
/// <param name="numOfTrialsPerLevel">Number of trails to run per gaussian level</param>
void runNoiseTrials(int numOfTrialsPerLevel)
{
    const float FOCAL_LEN = 600.f;   // Ground truth focal length
    const float noiseLevels[] = { 0, 0.01f, 0.1f, 0.5f, 1, 2 };
    const String noiseLevelNames[] = { "0", "0_01", "0_1", "0_5", "1", "2" };
    const int numOfLevels = sizeof(noiseLevels) / sizeof(noiseLevels[0]);
    const int NUM_OF_RANDOM_POINTS = 1000;

    const std::string resultDir = "results";    
    Mat intrinsicMatrix = createIntrinsicMatrix(FOCAL_LEN);

    for (int noiseLevelIndex = 0; noiseLevelIndex < numOfLevels; noiseLevelIndex++)
    {
        vector<TestResult> results;
        string resultFilePath = resultDir + "/noise_trial_" + noiseLevelNames[noiseLevelIndex] + ".csv";
        for (int trialIndex = 0; trialIndex < numOfTrialsPerLevel; trialIndex++)
        {
            RNG rng((noiseLevelIndex * numOfTrialsPerLevel) + trialIndex);
            TestResult result;

            // First generate all of our 3D points
            vector<Point3f> points3d = getRandom3Dpoints(NUM_OF_RANDOM_POINTS);
            vector<Point3f> noisyPoints3d = addNoiseToPoints(points3d, noiseLevels[noiseLevelIndex]);

            // Generate the random rotation
            Mat rotationMat;
            getRandom3DRotationMat(rotationMat);

            // Convert rotation and translation vectors
            Mat rotationVector, translationVector;
            getRotationTranslationVectorsFromRotationMat(rotationMat, rotationVector, translationVector);

            // Get four point estimations
            Mat homogeneousP1, homogeneousP2;
            getHomogeneousMatsFrom3DPoints(points3d, noisyPoints3d, intrinsicMatrix, rotationVector, translationVector, homogeneousP1, homogeneousP2);

            // Estimate the fundamental matrix with each solver
            vector<TestResult> solverResults = estimateFundamentals(trialIndex, homogeneousP1, homogeneousP2, intrinsicMatrix, rotationMat, translationVector);
            results.insert(results.end(), solverResults.begin(), solverResults.end());
        }
        saveTrailResults(resultFilePath, results);
    }
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
    const std::string resultDir = "results";
    Mat intrinsicMatrix = createIntrinsicMatrix(FOCAL_LEN);

    vector<double> errorNorms;
    vector<TestResult> results;

    for (int i = 0; i < numOfTrials; i++)
    {
        RNG rng(i);

        TestResult result;

        // First generate all of our 3D points
        vector<Point3f> points3d = getRandom3Dpoints(NUM_OF_RANDOM_POINTS);

        // Generate the random rotation
        Mat rotationMat;
        getRandom3DRotationMat(rotationMat);

        // Convert rotation and translation vectors
        Mat rotationVector, translationVector;
        getRotationTranslationVectorsFromRotationMat(rotationMat, rotationVector, translationVector);

        // Get four point estimations
        Mat homogeneousP1, homogeneousP2;
        getHomogeneousMatsFrom3DPoints(points3d, intrinsicMatrix, rotationVector, translationVector, homogeneousP1, homogeneousP2);

        // Estimate the fundamental matrix with each solver
        vector<TestResult> solverResults = estimateFundamentals(i, homogeneousP1, homogeneousP2, intrinsicMatrix, rotationMat, translationVector);
        results.insert(results.end(), solverResults.begin(), solverResults.end());
    }

    saveTrailResults(resultDir + "/zero_noise_trail.csv", results);
}