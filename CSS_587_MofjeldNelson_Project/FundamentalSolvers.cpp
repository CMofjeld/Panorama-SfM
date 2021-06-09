/*
* CSS 587 - Advance Computer Vision
* Spring 2021
* Carl Mofjeld, Drew Nelson
*
* Description:
* Contains the methods compute the fundamental matrix using different solvers
*/

#include <Eigen/Eigenvalues>
#include <opencv2/core/eigen.hpp>
#include "FundamentalSolvers.h"
#include <unordered_set>
#include <opencv2/features2d.hpp>
#include <opencv2/sfm/fundamental.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>
using namespace std;
using namespace cv;

/// <summary>
/// Reconstruct a spherical motion fundamental matrix from its vectorized equivalent
/// </summary>
/// <param name="f">Vectorized version of the fundamental matrix</param>
/// <returns>Reconstructed fundamental matrix</returns>
Mat reconstructFundamentalFromVector(const Vec6f& f) {
   Mat F = (Mat_<float>(3, 3) << 
      f(0), f(1), f(2),
      f(1), -f(0), f(3),
      f(4), f(5), 0);
   return F;
}

/// <summary>
/// Construct design matrix for linear system resulting from the epipolar constraint
/// for spherical camera motion and no radial distortion.
/// </summary>
/// <param name="points1">Nx3 matrix of homogeneous points in the first image</param>
/// <param name="points2">Nx3 matrix of homogeneous points in the second image</param>
/// <returns>Design matrix</returns>
bool matsAreSameSize(const Mat& mat1, const Mat& mat2) {
    return ((mat1.rows == mat2.rows) && (mat1.cols == mat2.cols));
}

/// <summary>
/// Construct design matrix for linear system resulting from the epipolar constraint
/// for spherical camera motion and no radial distortion.
/// </summary>
/// <param name="points1">Nx3 matrix of homogeneous points in the first image</param>
/// <param name="points2">Nx3 matrix of homogeneous points in the second image</param>
/// <returns>Design matrix</returns>
Mat getDesignMatrixFromPoints(const Mat& points1, const Mat& points2) {
   // Input validation
   CV_Assert(matsAreSameSize(points1, points2));

   size_t num_correspondences = points1.rows;
   Mat A(num_correspondences, 6, CV_32F);
   Mat x1 = points1.col(0);
   Mat y1 = points1.col(1);
   Mat x2 = points2.col(0);
   Mat y2 = points2.col(1);

   A.col(0) = x1.mul(x2) - y1.mul(y2);
   A.col(1) = x1.mul(y2) + y1.mul(x2);
   x2.copyTo(A.col(2));
   y2.copyTo(A.col(3));
   x1.copyTo(A.col(4));
   y1.copyTo(A.col(5));

   return A;
}

/// <summary>
/// Estimate the fundamental matrix between two images using four point correspondences.
/// </summary>
/// <param name="points1">List of four points in the first image</param>
/// <param name="points2">List of four points in the second image</param>
/// <param name="solutions">[Output] Possible solutions for the fundamental matrix</param>
void fourPointMethod(const Mat& points1, const Mat& points2, vector<Mat>& solutions) {
   // Input validation
   CV_Assert(points1.rows >= 4);
   CV_Assert(matsAreSameSize(points1, points2));

   // Set up system of linear equations based on the epipolar constraint
   Mat A = getDesignMatrixFromPoints(points1, points2);

   // Find a basis for the nullspace (and the vectorized fundamental matrix, Fv)
   SVD svd(A, SVD::FULL_UV);  // FULL_UV flag needed to get nullspace basis vectors

   // Fv is a combination of the nullspace vectors according to the equation:
   // Fv = x*F1 + (1-x)*F2
   // where x is scalar determining the relative proportion of each nullspace vector.
   Vec6f f1 = svd.vt.row(4); // first nullspace basis vector
   Vec6f f2 = svd.vt.row(5); // second nullspace basis vector
   vector<Mat> Fmats(2);   // fundamental matrices generated from the basis vectors
   Fmats[0] = reconstructFundamentalFromVector(f1).t();
   Fmats[1] = reconstructFundamentalFromVector(f2).t();

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
/// Estimate the fundamental matrix between two images using six point correspondences.
/// </summary>
/// <param name="points1">Nx3 matrix of homogeneous points in the first image</param>
/// <param name="points2">Nx3 matrix of homogeneous points in the second image</param>
/// <param name="solutions">[Output] Possible solutions for the fundamental matrix</param>
void sixPointMethod(const Mat& points1, const Mat& points2, vector<Mat>& solutions) {
   // Input validation
   CV_Assert(points1.rows >= 6);
   CV_Assert(matsAreSameSize(points1, points2));

   // Construct the design matrices for the equation: (lambda*C1 + C2)f = 0
   // where f is the vectorized fundamental matrix and lambda is the radial distortion
   size_t num_correspondences = points1.rows;
   Mat C1(num_correspondences, 6, CV_32F);

   // Extract x-coords, y-coords, and squared lengths for each set of points
   Mat x1 = points1.col(0);
   Mat y1 = points1.col(1);
   Mat x2 = points2.col(0);
   Mat y2 = points2.col(1);
   Mat r1 = x1.mul(x1) + y1.mul(y1);
   Mat r2 = x2.mul(x2) + y2.mul(y2);

   // Construct C1
   C1.col(0).setTo(Scalar(0.f));
   C1.col(1).setTo(Scalar(0.f));
   C1.col(2) = x2.mul(r1);
   C1.col(3) = y2.mul(r1);
   C1.col(4) = x1.mul(r2);
   C1.col(5) = y1.mul(r2);

   // Construct C2
   // Note that C2 is the same design matrix that the 4-point method uses.
   Mat C2 = getDesignMatrixFromPoints(points1, points2);

   // Solve the generalized eigenvalue problem
   Eigen::MatrixXf C1_eigen, C2_eigen;
   cv2eigen(C1, C1_eigen);
   cv2eigen(C2, C2_eigen);
   Eigen::GeneralizedEigenSolver<Eigen::MatrixXf> esolver;
   esolver.compute(C2_eigen, C1_eigen);

   // The real eigenvalues are potential solutions for lambda
   for (int i = 0; i < esolver.eigenvalues().size(); i++)
   {
      if (esolver.eigenvalues()(i).imag() == 0) {
         float lambda = esolver.eigenvalues()(i).real();
         Eigen::VectorXf f_eigen = esolver.eigenvectors().col(i).real();
         Vec6f f;
         eigen2cv(f_eigen, f);
         Mat solution = reconstructFundamentalFromVector(f);
         solutions.push_back(solution);
      }
   }
}

/// <summary>
/// Estimate the fundamental matrix between two images using seven point correspondences.
/// </summary>
/// <param name="points1">Nx3 matrix of homogeneous points in the first image</param>
/// <param name="points2">Nx3 matrix of homogeneous points in the second image</param>
/// <param name="solutions">[Output] Possible solutions for the fundamental matrix</param>
void sevenPointMethod(const Mat& points1, const Mat& points2, vector<Mat>& solutions) {
   // Input validation
   CV_Assert(points1.rows >= 7);
   CV_Assert(matsAreSameSize(points1, points2));

   Mat solver_results = findFundamentalMat(points1, points2, FM_7POINT);
   solver_results.convertTo(solver_results, CV_32F);
   for (size_t i = 0; i < solver_results.rows; i += 3)
   {
      Mat single_result = solver_results(Rect(0, i, 3, 3));
      solutions.push_back(single_result / norm(single_result));
   }
}

/// <summary>
/// Estimate the fundamental matrix between two images using eight point correspondences.
/// </summary>
/// <param name="points1">Nx3 matrix of homogeneous points in the first image</param>
/// <param name="points2">Nx3 matrix of homogeneous points in the second image</param>
/// <param name="solutions">[Output] Possible solutions for the fundamental matrix</param>
void eightPointMethod(const Mat& points1, const Mat& points2, vector<Mat>& solutions) {
   // Input validation
   CV_Assert(points1.rows >= 8);
   CV_Assert(matsAreSameSize(points1, points2));

    Mat solver_results = findFundamentalMat(points1, points2, FM_8POINT);
    solver_results.convertTo(solver_results, CV_32F);
    for (size_t i = 0; i < solver_results.rows; i += 3)
    {
        Mat single_result = solver_results(Rect(0, i, 3, 3));
        solutions.push_back(single_result / norm(single_result));
    }
}

/// <summary>
/// Estimate the fundamental matrix robustly with RANSAC and the four point method.
/// </summary>
/// <param name="solver">Details which solver we should use to estimate the fundamental matrix</param>
/// <param name="points1">Nx3 matrix of homogeneous points in the first image</param>
/// <param name="points2">Nx3 matrix of homogeneous points in the second image</param>
/// <param name="iterations">Maximum number of iterations</param>
/// <param name="threshold">Error threshold for inlier/outlier calculation</param>
/// <returns>Estimated fundamental matrix</returns>
Mat estimateFundamentalMatrix(CustomSolver solver, const Mat& points1, const Mat& points2, int iterations, float threshold)
{
   // Input validation
   CV_Assert(matsAreSameSize(points1, points2));

   // Setup
   Mat bestEstimate;       // Best estimate for fundamental matrix
   int bestInliers = -1;   // Number of inliers for best estimate
   static RNG rng(12345);  // Random number generator for selecting random subsets
   int minIndex = 0;                // Minimum subset index
   int maxIndex = points1.rows;     // Maximum subset index

   // For specified iterations:
   for (int i = 0; i < iterations; i++)
   {
      // Select random subsample of 4
      unordered_set<int> previousIndices;
      Mat subsample1(solver.requiredNumOfPoints, points1.cols, CV_32F), subsample2(solver.requiredNumOfPoints, points2.cols, CV_32F);
      for (size_t i = 0; i < solver.requiredNumOfPoints; i++)
      {
         int newRandomIndex;
         do
         {
            newRandomIndex = rng.uniform(minIndex, maxIndex);
         } while (previousIndices.count(newRandomIndex) > 0);
         previousIndices.insert(newRandomIndex);
         points1.row(newRandomIndex).copyTo(subsample1.row(i));
         points2.row(newRandomIndex).copyTo(subsample2.row(i));
      }

      // Get potential solutions
      vector<Mat> solutions;

      switch (solver.solverType)
      {
          case SolverType::FourPoint:
              fourPointMethod(subsample1, subsample2, solutions);
              break;
          case SolverType::SixPoint:
              sixPointMethod(subsample1, subsample2, solutions);
              break;
          case SolverType::CV_SevenPoint:
              sevenPointMethod(subsample1, subsample2, solutions);
              break;
          case SolverType::CV_EightPoint:
              eightPointMethod(subsample1, subsample2, solutions);
              break;
      }

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
/// <param name="points1">Nx3 matrix of homogeneous points in the first image</param>
/// <param name="points2">Nx3 matrix of homogeneous points in the second image</param>
/// <param name="threshold">Error threshold for inlier/outlier calculation</param>
/// <returns>Number of inliers</returns>
int countInliersFundamental(const Mat& F, const Mat& points1, const Mat& points2, float threshold) {
   // Input validation
   CV_Assert(matsAreSameSize(points1, points2));

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
/// Estimate the fundamental matrix between two image pairs using SIFT descriptors
/// and the four point estimator.
/// </summary>
/// <param name="img1">First image</param>
/// <param name="img2">Second image</param>
/// <returns>Estimated fundamental matrix</returns>
Mat fundamentalFromImagePair(const Mat& img1, const Mat& img2) {
   // Create a SIFT detector/descriptor
   Ptr<SIFT> detector = SIFT::create();

   // Use the detectAndCompute method of the detector to obtain both keypoints
   // and descriptors from both input images. 
   vector<KeyPoint> keypoints1, keypoints2;
   Mat descriptors1, descriptors2;
   detector->detectAndCompute(img1, Mat(), keypoints1, descriptors1);
   detector->detectAndCompute(img2, Mat(), keypoints2, descriptors2);

   // Create a brute-force matcher
   Ptr<BFMatcher> matcher = BFMatcher::create();

   // Use the brute-force matcher to compute matches between the
   // keypoints/descriptors in the two images.
   vector<DMatch> matches;
   matcher->match(descriptors1, descriptors2, matches);

   // Create vectors of matched points
   vector<Point2f> matches1, matches2;
   for (auto& match : matches) {
      matches1.push_back(keypoints1[match.queryIdx].pt);
      matches2.push_back(keypoints2[match.trainIdx].pt);
   }

   // Convert to homogeneous matrices
   Mat homogeneousP1, homogeneousP2;
   hconcat(Mat(matches1.size(), 2, CV_32F, matches1.data()), Mat::ones(matches1.size(), 1, CV_32F), homogeneousP1);
   hconcat(Mat(matches2.size(), 2, CV_32F, matches2.data()), Mat::ones(matches2.size(), 1, CV_32F), homogeneousP2);

   // Estimate the fundamental matrix
   return estimateFundamentalMatrix(FourPointSolver, homogeneousP1, homogeneousP2);
}

/// <summary>
/// Estimate the fundamental matrix between two image pairs using SIFT descriptors
/// and the four point estimator.
/// 
/// NOTE: This is similar the method above, but was added to help debug the issue with reconstruction. The main difference
/// with this method is that it returns out the point matches in each image
/// </summary>
/// <param name="img1">First image</param>
/// <param name="img2">Second image</param>
/// <param name="matches1">Points found in image 1</param>
/// <param name="matches2">Points found in image 2</param>
/// <returns>Estimated fundamental matrix</returns>
Mat fundamentalFromImagePair(const Mat& img1, const Mat& img2, vector<Point2f> &matches1, vector<Point2f> &matches2) {
    // Create a SIFT detector/descriptor
    Ptr<SIFT> detector = SIFT::create();

    // Use the detectAndCompute method of the detector to obtain both keypoints
    // and descriptors from both input images. 
    vector<KeyPoint> keypoints1, keypoints2;
    Mat descriptors1, descriptors2;
    detector->detectAndCompute(img1, Mat(), keypoints1, descriptors1);
    detector->detectAndCompute(img2, Mat(), keypoints2, descriptors2);

    // Create a brute-force matcher
    Ptr<BFMatcher> matcher = BFMatcher::create();

    // Use the brute-force matcher to compute matches between the
    // keypoints/descriptors in the two images.
    vector<DMatch> matches;
    matcher->match(descriptors1, descriptors2, matches);

    // Create vectors of matched points
    for (auto& match : matches) {
        matches1.push_back(keypoints1[match.queryIdx].pt);
        matches2.push_back(keypoints2[match.trainIdx].pt);
    }

    // Convert to homogeneous matrices
    Mat homogeneousP1, homogeneousP2;
    hconcat(Mat(matches1.size(), 2, CV_32F, matches1.data()), Mat::ones(matches1.size(), 1, CV_32F), homogeneousP1);
    hconcat(Mat(matches2.size(), 2, CV_32F, matches2.data()), Mat::ones(matches2.size(), 1, CV_32F), homogeneousP2);

    // Estimate the fundamental matrix
    return estimateFundamentalMatrix(FourPointSolver, homogeneousP1, homogeneousP2);
}

/// <summary>
/// Decompose a Fundamental matrix into a relative rotation and translation that are
/// consistent with outward facing spherical motion.
/// </summary>
/// <param name="F">Fundamental matrix</param>
/// <param name="K">Camera intrinsic matrix</param>
/// <param name="R">[Output] Rotation matrix</param>
/// <param name="t">[Output] Translation vector</param>
/// <returns>Estimated fundamental matrix</returns>
void decomposeFundamentalMat(const Mat& F, const Mat& K, Mat& R, Mat& t) {
   // First get the essential matrix from the fundamental
   Mat E;   // Essential matrix
   sfm::essentialFromFundamental(F, K, K, E);

   // Decompose the essential matrix
   Mat Ra;  // first rotation estimate
   Mat Rb;  // second rotation estimate
   decomposeEssentialMat(E, Ra, Rb, t);

   // Determine which estimate for rotation is consistent with spherical motion
   Mat ta;  // translation according to Ra
   Mat tb;  // translation according to Rb
   Ra.col(2).copyTo(ta);
   ta.at<float>(2, 0) -= 1;
   Rb.col(2).copyTo(tb);
   tb.at<float>(2, 0) -= 1;
   double sa = ta.dot(t) / norm(ta);   // score first estimate
   double sb = tb.dot(t) / norm(tb);   // score second estimate
   R = sa > sb ? Ra : Rb;
}