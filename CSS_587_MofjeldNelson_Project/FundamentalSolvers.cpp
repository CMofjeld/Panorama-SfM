#include "FundamentalSolvers.h"
#include <opencv2/calib3d.hpp>
//#include <opencv2/sfm/fundamental.hpp> TODO: install SFM module and uncomment this include
#include <iostream>

/// <summary>
/// Estimate the fundamental matrix between two images using four point correspondences.
/// </summary>
/// <param name="points1">List of four points in the first image</param>
/// <param name="points2">List of four points in the second image</param>
/// <returns>Estimated fundamental matrix</returns>
Mat fourPointMethod(const vector<Point2f>& points1, const vector<Point2f>& points2) {
   // Set up system of linear equations based on the epipolar constraint

   // Find a basis for the nullspace (and the vectorized fundamental matrix)

   // The fundamental matrix must have a zero determinant.
   // Set up a cubic equation based on this to find the relative proportions
   // of the two basis vectors that make up F.

   // Reconstruct F based on the results above
   Mat F = Mat::zeros(Size(3, 3), CV_32FC1);
   return F;
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
   cout << "Random 3D points:" << endl; //DEBUG
   for (auto& point : points3d) cout << point << endl; // DEBUG
   cout << endl; //DEBUG

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
   cout << "Camera intrinsics matrix: " << endl; // DEBUG
   cout << K << endl << endl; // DEBUG

   Mat R; // Rotation matrix
   getRandom3DRotationMat(R, MAX_ROTATION, MAX_ROTATION, MAX_ROTATION);
   Mat rvec;   // Rotation vector
   Rodrigues(R, rvec);
   cout << "Rotation matrix:" << endl; // DEBUG
   cout << R << endl << endl; // DEBUG
   cout << "Rotation vector:" << endl; // DEBUG
   cout << rvec << endl << endl; // DEBUG
   Mat t; // Translation vector
   R.col(2).copyTo(t);
   t.at<float>(2, 0) -= 1;
   cout << "Translation vector:" << endl; // DEBUG
   cout << t << endl << endl; // DEBUG

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

   // Get four point estimation
   Mat fourPointEstimate = fourPointMethod(projectedPoints1, projectedPoints2);

   // Compare to ground truth
   Mat F;
   fundamentalFromKRT(F, K, K, R, t);
   cout << "Ground truth Fundamental:" << endl; //DEBUG
   cout << F << endl << endl; //DEBUG
   cout << "Estimated Fundamental:" << endl; //DEBUG
   cout << fourPointEstimate << endl << endl; //DEBUG
   cout << "Frobeniums norm of error:" << endl; //DEBUG
   cout << norm(F - fourPointEstimate) << endl << endl; //DEBUG
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
   RNG rng(12345);
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
/// Generate 3D rotation matrix with random rotations selected from a uniform distribution.
/// </summary>
/// <param name="rotationMat">[Output] Rotation matrix</param>
/// <param name="xRotMax">Maximum magnitude of rotation around x-axis (radians)</param>
/// <param name="yRotMax">Maximum magnitude of rotation around y-axis (radians)</param>
/// <param name="zRotMax">Maximum magnitude of rotation around z-axis (radians)</param>
void getRandom3DRotationMat(Mat& rotationMat, float xRotMax, float yRotMax, float zRotMax) {
   RNG rng(12345);
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
      cos(x_angle), -sin(x_angle), 0,
      sin(x_angle), cos(x_angle), 0,
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
   //TODO: install SFM module and uncomment this code block
   //// Get matrix representation of cross-product with t
   //Mat t_x = (Mat_<float>(3, 3) <<
   //   0, -t.at<float>(2, 0), t.at<float>(1, 0),
   //   t.at<float>(2, 0), 0, -t.at<float>(0, 0),
   //   -t.at<float>(1, 0), t.at<float>(0, 0), 0);
   //Mat E = t_x * R;
   //fundamentalFromEssential(E, K1, K2, F);

   //TODO: install SFM module and remove this code block
   // Reference for this code: https://sourishghosh.com/2016/fundamental-matrix-from-camera-matrices/
   Mat A = K1 * R.t() * t;
   Mat C = (Mat_<float>(3, 3) <<
         0, -A.at<float>(2, 0), A.at<float>(1, 0),
         A.at<float>(2, 0), 0, -A.at<float>(0, 0),
         -A.at<float>(1, 0), A.at<float>(0, 0), 0);
   F = (K2.inv()).t() * R * K1.t() * C;
}