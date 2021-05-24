#include "FundamentalSolvers.h"
#include <opencv2/calib3d.hpp>
//#include <opencv2/sfm/fundamental.hpp> TODO: install SFM module and uncomment this include
#include <iostream>
#include <limits>
#include <unordered_set>

/// <summary>
/// Estimate the fundamental matrix between two images using four point correspondences.
/// </summary>
/// <param name="points1">List of four points in the first image</param>
/// <param name="points2">List of four points in the second image</param>
/// <param name="solutions">[Output] Possible solutions for the fundamental matrix</param>
/// <returns>Estimated fundamental matrix</returns>
void fourPointMethod(const vector<Point2f>& points1, const vector<Point2f>& points2, vector<Mat>& solutions) {
   // TODO: input validation
   
   // Set up system of linear equations based on the epipolar constraint
   size_t num_correspondences = points1.size();
   Mat A(num_correspondences, 6, CV_32FC1);
   for (size_t i = 0; i < num_correspondences; i++)
   {
      A.at<float>(i, 0) = points1[i].x * points2[i].x - points1[i].y * points2[i].y;
      A.at<float>(i, 1) = points1[i].x * points2[i].y + points1[i].y * points2[i].x;
      A.at<float>(i, 2) = points2[i].x;
      A.at<float>(i, 3) = points2[i].y;
      A.at<float>(i, 4) = points1[i].x;
      A.at<float>(i, 5) = points1[i].y;
   }

   // Find a basis for the nullspace (and the vectorized fundamental matrix, Fv)
   SVD svd(A, SVD::FULL_UV);  // FULL_UV flag needed to get nullspace basis vectors

   // Fv is a combination of the nullspace vectors according to the equation:
   // Fv = x*F1 + (1-x)*F2
   // where x is scalar determining the relative proportion of each nullspace vector.
   Vec6f f1 = svd.vt.row(4); // first nullspace basis vector
   Vec6f f2 = svd.vt.row(5); // second nullspace basis vector

   vector<Mat> Fmats(2);   // fundamental matrices generated from the basis vectors
   Fmats[0] = (Mat_<float>(3, 3) << f1(0), f1(1),  f1(4),
               f1(1), -f1(0), f1(5),
               f1(2), f1(3),  0);
   Fmats[1] = (Mat_<float>(3, 3) << f2(0), f2(1),  f2(4),
               f2(1), -f2(0), f2(5),
               f2(2), f2(3),  0);

   // The fundamental matrix must have a zero determinant.
   // Set up a cubic equation based on this to find x.
   // Reference for this code:
   // https://imkaywu.github.io/blog/2017/06/fundamental-matrix/
   float D[2][2][2]; // intermediate determinants
   for (size_t i1 = 0; i1 < 2; i1++)
   {
      for (size_t i2 = 0; i2 < 2; i2++)
      {
         for (size_t i3 = 0; i3 < 2; i3++)
         {
            Mat_<float> Dtmp(3,3);
            Fmats[i1].col(0).copyTo(Dtmp.col(0));
            Fmats[i2].col(1).copyTo(Dtmp.col(1));
            Fmats[i3].col(2).copyTo(Dtmp.col(2));
            D[i1][i2][i3] = determinant(Dtmp);
         }
      }
   }
   Vec4f coefficients(4);  // coefficients of the cubic equation
   coefficients(0) = -D[1][0][0] + D[0][1][1] + D[0][0][0] + D[1][1][0] + D[1][0][1] - D[0][1][0] - D[0][0][1] - D[1][1][1];
   coefficients(1) = D[0][0][1] - 2 * D[0][1][1] - 2 * D[1][0][1] + D[1][0][0] - 2 * D[1][1][0] + D[0][1][0] + 3 * D[1][1][1];
   coefficients(2) = D[1][1][0] + D[0][1][1] + D[1][0][1] - 3 * D[1][1][1];
   coefficients(3) = D[1][1][1];
   Vec3f roots;    // solutions to the cubic equation
   int numRoots = solveCubic(coefficients, roots);

   // The cubic equation may have up to 3 possible solutions for x.
   // Reconstruct a possible fundamental matrix solution for each x
   // and store it in the output vector.
   for (size_t i = 0; i < numRoots; i++)
   {
      Mat solution = Fmats[0] * roots[i] + Fmats[1] * (1.f - roots[i]);
      solutions.push_back(solution.t() / norm(solution));
   }
}

/// <summary>
/// Estimate the fundamental matrix between two images using four point correspondences.
/// </summary>
/// <param name="points1">List of four points in the first image</param>
/// <param name="points2">List of four points in the second image</param>
/// <param name="solutions">[Output] Possible solutions for the fundamental matrix</param>
void fourPointMethod(const Mat& points1, const Mat& points2, vector<Mat>& solutions) {
   // TODO: input validation

   // Set up system of linear equations based on the epipolar constraint
   size_t num_correspondences = points1.rows;
   Mat A(num_correspondences, 6, CV_32F);
   A.col(0) = points1.col(0).mul(points2.col(0)) - points1.col(1).mul(points2.col(1));
   A.col(1) = points1.col(0).mul(points2.col(1)) + points1.col(1).mul(points2.col(0));
   points2.col(0).copyTo(A.col(2));
   points2.col(1).copyTo(A.col(3));
   points1.col(0).copyTo(A.col(4));
   points1.col(1).copyTo(A.col(5));

   // Find a basis for the nullspace (and the vectorized fundamental matrix, Fv)
   SVD svd(A, SVD::FULL_UV);  // FULL_UV flag needed to get nullspace basis vectors

   // Fv is a combination of the nullspace vectors according to the equation:
   // Fv = x*F1 + (1-x)*F2
   // where x is scalar determining the relative proportion of each nullspace vector.
   Vec6f f1 = svd.vt.row(4); // first nullspace basis vector
   Vec6f f2 = svd.vt.row(5); // second nullspace basis vector

   vector<Mat> Fmats(2);   // fundamental matrices generated from the basis vectors
   Fmats[0] = (Mat_<float>(3, 3) << f1(0), f1(1), f1(4),
      f1(1), -f1(0), f1(5),
      f1(2), f1(3), 0);
   Fmats[1] = (Mat_<float>(3, 3) << f2(0), f2(1), f2(4),
      f2(1), -f2(0), f2(5),
      f2(2), f2(3), 0);

   // The fundamental matrix must have a zero determinant.
   // Set up a cubic equation based on this to find x.
   // Reference for this code:
   // https://imkaywu.github.io/blog/2017/06/fundamental-matrix/
   float D[2][2][2]; // intermediate determinants
   for (size_t i1 = 0; i1 < 2; i1++)
   {
      for (size_t i2 = 0; i2 < 2; i2++)
      {
         for (size_t i3 = 0; i3 < 2; i3++)
         {
            Mat_<float> Dtmp(3, 3);
            Fmats[i1].col(0).copyTo(Dtmp.col(0));
            Fmats[i2].col(1).copyTo(Dtmp.col(1));
            Fmats[i3].col(2).copyTo(Dtmp.col(2));
            D[i1][i2][i3] = determinant(Dtmp);
         }
      }
   }
   Vec4f coefficients(4);  // coefficients of the cubic equation
   coefficients(0) = -D[1][0][0] + D[0][1][1] + D[0][0][0] + D[1][1][0] + D[1][0][1] - D[0][1][0] - D[0][0][1] - D[1][1][1];
   coefficients(1) = D[0][0][1] - 2 * D[0][1][1] - 2 * D[1][0][1] + D[1][0][0] - 2 * D[1][1][0] + D[0][1][0] + 3 * D[1][1][1];
   coefficients(2) = D[1][1][0] + D[0][1][1] + D[1][0][1] - 3 * D[1][1][1];
   coefficients(3) = D[1][1][1];
   Vec3f roots;    // solutions to the cubic equation
   int numRoots = solveCubic(coefficients, roots);

   // The cubic equation may have up to 3 possible solutions for x.
   // Reconstruct a possible fundamental matrix solution for each x
   // and store it in the output vector.
   for (size_t i = 0; i < numRoots; i++)
   {
      Mat solution = Fmats[0] * roots[i] + Fmats[1] * (1.f - roots[i]);
      solutions.push_back(solution.t() / norm(solution));
   }
}

/// <summary>
/// Estimate the fundamental matrix robustly with RANSAC and the four point method.
/// </summary>
/// <param name="points1">List of points in the first image</param>
/// <param name="points2">List of points in the second image</param>
/// <param name="iterations">Maximum number of iterations</param>
/// <param name="threshold">Error threshold for inlier/outlier calculation</param>
/// <returns>Estimated fundamental matrix</returns>
Mat estimateFundamentalMatrix(const vector<Point2f>& points1, const vector<Point2f>& points2, int iterations = 500, float threshold = 1.0)
{
   // TODO: input validation
   Mat bestEstimate;       // Best estimate for fundamental matrix
   int bestInliers = -1;   // Number of inliers for best estimate
   static RNG rng(12345);  // Random number generator for selecting random subsets
   int minIndex = 0;                // Minimum subset index
   int maxIndex = points1.size();   // Maximum subset index
   const int subsampleSize = 4;     // Minimum number of correspondences to estimate F

   // For specified iterations:
   for (int i = 0; i < iterations; i++)
   {
      // Select random subsample of 4
      vector<int> indices;
      unordered_set<int> previousIndices;
      for (size_t i = 0; i < subsampleSize; i++)
      {
         int newRandomIndex;
         do
         {
            newRandomIndex = rng.uniform(minIndex, maxIndex);
         } while (previousIndices.count(newRandomIndex) > 0);
         indices.push_back(newRandomIndex);
      }
      vector<Point2f> subsample1, subsample2;
      for (auto& index : indices) {
         subsample1.push_back(points1[index]);
         subsample2.push_back(points2[index]);
      }

      // Get potential solutions
      vector<Mat> solutions;
      fourPointMethod(subsample1, subsample2, solutions);

      // Check for new best estimate
      for (auto& solution : solutions) {
         int curInliers = countInliersFundamental(solution, points1, points2, threshold);
         if (curInliers > bestInliers) {
            bestEstimate = solution;
            bestInliers = curInliers;
         }
      }
   }
   // Return best estimate
   return bestEstimate;
}

/// <summary>
/// Estimate the fundamental matrix robustly with RANSAC and the four point method.
/// </summary>
/// <param name="points1">Nx3 matrix of homogeneous points in the first image</param>
/// <param name="points2">Nx3 matrix of homogeneous points in the second image</param>
/// <param name="iterations">Maximum number of iterations</param>
/// <param name="threshold">Error threshold for inlier/outlier calculation</param>
/// <returns>Estimated fundamental matrix</returns>
Mat estimateFundamentalMatrix(const Mat& points1, const Mat& points2, int iterations = 500, float threshold = 1.0)
{
   // TODO: input validation
   Mat bestEstimate;       // Best estimate for fundamental matrix
   int bestInliers = -1;   // Number of inliers for best estimate
   static RNG rng(12345);  // Random number generator for selecting random subsets
   int minIndex = 0;                // Minimum subset index
   int maxIndex = points1.rows;     // Maximum subset index
   const int subsampleSize = 4;     // Minimum number of correspondences to estimate F

   // For specified iterations:
   for (int i = 0; i < iterations; i++)
   {
      // Select random subsample of 4
      unordered_set<int> previousIndices;
      Mat subsample1(subsampleSize, points1.cols, CV_32F), subsample2(subsampleSize, points2.cols, CV_32F);
      for (size_t i = 0; i < subsampleSize; i++)
      {
         int newRandomIndex;
         do
         {
            newRandomIndex = rng.uniform(minIndex, maxIndex);
         } while (previousIndices.count(newRandomIndex) > 0);
         points1.row(newRandomIndex).copyTo(subsample1.row(i));
         points2.row(newRandomIndex).copyTo(subsample2.row(i));
      }

      // Get potential solutions
      vector<Mat> solutions;
      fourPointMethod(subsample1, subsample2, solutions);

      // Check for new best estimate
      for (auto& solution : solutions) {
         int curInliers = countInliersFundamental(solution, points1, points2, threshold);
         if (curInliers > bestInliers) {
            bestEstimate = solution;
            bestInliers = curInliers;
         }
      }
   }
   // Return best estimate
   return bestEstimate;
}

/// <summary>
/// Count the number of inliers in a set of point correspondences based on a given
/// Fundamental matrix and a threshold for error in the epipolar constraint.
/// </summary>
/// <param name="F">Fundamental matrix</param>
/// <param name="points1">List of points in the first image</param>
/// <param name="points2">List of points in the second image</param>
/// <param name="threshold">Error threshold for inlier/outlier calculation</param>
/// <returns>Number of inliers</returns>
int countInliersFundamental(const Mat& F, const vector<Point2f>& points1, const vector<Point2f>& points2, float threshold) {
   // Convert to matrices for quicker calculation
   Mat homogeneousP1, homogeneousP2;
   vector<Point2f> points1copy(points1), points2copy(points2);
   hconcat(Mat(points1copy.size(), 2, CV_32F, points1copy.data()), Mat::ones(points1copy.size(), 1, CV_32F), homogeneousP1);
   hconcat(Mat(points2copy.size(), 2, CV_32F, points2copy.data()), Mat::ones(points2copy.size(), 1, CV_32F), homogeneousP2);
   
   // Get epipolar lines
   Mat epipolarLines = homogeneousP2 * F;

   // Count inliers
   int inliers = 0;
   for (size_t i = 0; i < epipolarLines.rows; i++)
   {
      double error = homogeneousP1.row(i).dot(epipolarLines.row(i));
      if (error < threshold) ++inliers;
   }
   return inliers;
}

/// <summary>
/// Count the number of inliers in a set of point correspondences based on a given
/// Fundamental matrix and a threshold for error in the epipolar constraint.
/// </summary>
/// <param name="F">Fundamental matrix</param>
/// <param name="points1">Nx3 matrix of homogeneous points in the first image</param>
/// <param name="points2">Nx3 matrix of homogeneous points in the second image</param>
/// <param name="threshold">Error threshold for inlier/outlier calculation</param>
/// <returns>Number of inliers</returns>
int countInliersFundamental(const Mat& F, const Mat& points1, const Mat& points2, float threshold) {
   // Get epipolar lines
   Mat epipolarLines = points2 * F;

   // Count inliers
   int inliers = 0;
   for (size_t i = 0; i < epipolarLines.rows; i++)
   {
      if (points1.row(i).dot(epipolarLines.row(i)) < threshold) ++inliers;
   }
   return inliers;
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