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
/// Estimate the fundamental matrix between two images using six point correspondences.
/// </summary>
/// <param name="points1">List of four points in the first image</param>
/// <param name="points2">List of four points in the second image</param>
/// <param name="solutions">[Output] Possible solutions for the fundamental matrix</param>
void sixPointMethod(const vector<Point2f>& points1, const vector<Point2f>& points2, vector<Mat>& solutions) {

}

/// <summary>
/// Estimate the fundamental matrix between two images using six point correspondences.
/// </summary>
/// <param name="points1">Nx3 matrix of homogeneous points in the first image</param>
/// <param name="points2">Nx3 matrix of homogeneous points in the second image</param>
/// <param name="solutions">[Output] Possible solutions for the fundamental matrix</param>
void sixPointMethod(const Mat& points1, const Mat& points2, vector<Mat>& solutions) {
   // TODO: input validation

   // Construct the design matrices for the equation: (lambda*C1 + C2)f = 0
   // where f is the vectorized fundamental matrix and lambda is the radial distortion
   size_t num_correspondences = points1.rows;
   Mat C1(num_correspondences, 6, CV_32F), C2(num_correspondences, 6, CV_32F);

   // Construct C1
   Mat r1 = points1.col(0).mul(points1.col(0)) + points1.col(1).mul(points1.col(1));
   Mat r2 = points2.col(0).mul(points2.col(0)) + points2.col(1).mul(points2.col(1));
   C1.col(0).setTo(Scalar(0.f));
   C1.col(1).setTo(Scalar(0.f));
   C1.col(2) = points2.col(1).mul(r1);
   C1.col(3) = points2.col(2).mul(r1);
   C1.col(4) = points1.col(1).mul(r2);
   C1.col(5) = points1.col(2).mul(r2);

   // Construct C2
   // Note that C2 is the same design matrix that the 4-point method uses.
   C2.col(0) = points1.col(0).mul(points2.col(0)) - points1.col(1).mul(points2.col(1));
   C2.col(1) = points1.col(0).mul(points2.col(1)) + points1.col(1).mul(points2.col(0));
   points2.col(0).copyTo(C2.col(2));
   points2.col(1).copyTo(C2.col(3));
   points1.col(0).copyTo(C2.col(4));
   points1.col(1).copyTo(C2.col(5));

   // Solve the generalized eigenvalue problem
   Eigen::MatrixXf C1_eigen, C2_eigen;
   cv2eigen(C1, C1_eigen);
   cv2eigen(C2, C2_eigen);
   Eigen::GeneralizedEigenSolver<Eigen::MatrixXf> esolver;
   esolver.compute(C2_eigen, C1_eigen);
   cout << "E-vals: " << endl << esolver.eigenvalues() << endl;
   cout << "E-vecs: " << endl << esolver.eigenvectors() << endl;

   // The real eigenvalues are potential solutions for lambda
   for (int i = 0; i < esolver.eigenvalues().size(); i++)
   {
      if (esolver.eigenvalues()(i).imag() == 0) {
         //cout << esolver.eigenvalues()(i).real() << endl;
         float lambda = esolver.eigenvalues()(i).real();
         Eigen::VectorXf f = esolver.eigenvectors().col(i).real();
         Eigen::Matrix3f F;
         F << f(0), f(1), f(2),
            f(1), -f(0), f(3),
            f(4), f(5), 0;
         Mat solution;
         eigen2cv(F, solution);
         solutions.push_back(solution);
      }
   }
}

/// <summary>
/// Estimate the fundamental matrix between two images using seven point correspondences.
/// </summary>
/// <param name="points1">List of seven points in the first image</param>
/// <param name="points2">List of seven points in the second image</param>
/// <param name="solutions">[Output] Possible solutions for the fundamental matrix</param>
void sevenPointMethod(const vector<Point2f>& points1, const vector<Point2f>& points2, vector<Mat>& solutions) {
   Mat solver_results = findFundamentalMat(points1, points2, FM_7POINT);
   solver_results.convertTo(solver_results, CV_32F);
   for (size_t i = 0; i < solver_results.rows; i += 3)
   {
      Mat single_result = solver_results(Rect(0, i, 3, 3));
      solutions.push_back(single_result / norm(single_result));
   }
}

/// <summary>
/// Estimate the fundamental matrix between two images using seven point correspondences.
/// </summary>
/// <param name="points1">Nx3 matrix of homogeneous points in the first image</param>
/// <param name="points2">Nx3 matrix of homogeneous points in the second image</param>
/// <param name="solutions">[Output] Possible solutions for the fundamental matrix</param>
void sevenPointMethod(const Mat& points1, const Mat& points2, vector<Mat>& solutions) {
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
/// <param name="points1">List of seven points in the first image</param>
/// <param name="points2">List of seven points in the second image</param>
/// <param name="solutions">[Output] Possible solutions for the fundamental matrix</param>
void eightPointMethod(const vector<Point2f>& points1, const vector<Point2f>& points2, vector<Mat>& solutions) {
    Mat solver_results = findFundamentalMat(points1, points2, FM_8POINT);
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
/// <param name="points1">List of points in the first image</param>
/// <param name="points2">List of points in the second image</param>
/// <param name="iterations">Maximum number of iterations</param>
/// <param name="threshold">Error threshold for inlier/outlier calculation</param>
/// <returns>Estimated fundamental matrix</returns>
Mat estimateFundamentalMatrix(const vector<Point2f>& points1, const vector<Point2f>& points2, int minimumPoints, int iterations, float threshold)
{
   // TODO: input validation
   Mat bestEstimate;       // Best estimate for fundamental matrix
   int bestInliers = -1;   // Number of inliers for best estimate
   static RNG rng(12345);  // Random number generator for selecting random subsets
   int minIndex = 0;                // Minimum subset index
   int maxIndex = points1.size();   // Maximum subset index
   const int SUBSAMPLE_SIZE = 4;    // Minimum points for estimating fundamental

   // For specified iterations:
   for (int i = 0; i < iterations; i++)
   {
      // Select random subsample of 4
      vector<int> indices;
      unordered_set<int> previousIndices;
      for (size_t i = 0; i < SUBSAMPLE_SIZE; i++)
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
      switch (minimumPoints)
      {
      case 4:
         fourPointMethod(subsample1, subsample2, solutions);
         break;
      case 7:
         sevenPointMethod(subsample1, subsample2, solutions);
         break;
      case 9:
         eightPointMethod(subsample1, subsample2, solutions);
         break;
      default:
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
   // TODO: input validation
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
         points1.row(newRandomIndex).copyTo(subsample1.row(i));
         points2.row(newRandomIndex).copyTo(subsample2.row(i));
      }

      // Get potential solutions
      vector<Mat> solutions;

      switch (solver.solverType)
      {
          case FourPoint:
              fourPointMethod(subsample1, subsample2, solutions);
              break;
          case CV_SevenPoint:
              sevenPointMethod(subsample1, subsample2, solutions);
              break;
          case CV_EightPoint:
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
   return estimateFundamentalMatrix(homogeneousP1, homogeneousP2);
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