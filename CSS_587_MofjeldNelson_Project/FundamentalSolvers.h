#pragma once
#include <opencv2/core.hpp>
#include <vector>

using namespace cv;
using namespace std;

enum SolverType
{
    //Unknown,
    FourPoint,
    SixPoint,
    CV_SevenPoint,
    CV_EightPoint
};

struct CustomSolver
{
    SolverType solverType;
    int requiredNumOfPoints;
};

const CustomSolver FourPointSolver = { FourPoint, 4 };
const CustomSolver SixPointSolver = { SixPoint, 6 };
const CustomSolver SevenPointSolver = { CV_SevenPoint, 7 };
const CustomSolver EightPointSolver = { CV_EightPoint, 8 };

/// <summary>
/// Estimate the fundamental matrix between two images using four point correspondences.
/// </summary>
/// <param name="points1">Nx3 matrix of homogeneous points in the first image</param>
/// <param name="points2">Nx3 matrix of homogeneous points in the second image</param>
/// <param name="solutions">[Output] Possible solutions for the fundamental matrix</param>
void fourPointMethod(const Mat& points1, const Mat& points2, vector<Mat>& solutions);

/// <summary>
/// Estimate the fundamental matrix between two images using six point correspondences.
/// </summary>
/// <param name="points1">Nx3 matrix of homogeneous points in the first image</param>
/// <param name="points2">Nx3 matrix of homogeneous points in the second image</param>
/// <param name="solutions">[Output] Possible solutions for the fundamental matrix</param>
void sixPointMethod(const Mat& points1, const Mat& points2, vector<Mat>& solutions);

/// <summary>
/// Estimate the fundamental matrix between two images using seven point correspondences.
/// </summary>
/// <param name="points1">Nx3 matrix of homogeneous points in the first image</param>
/// <param name="points2">Nx3 matrix of homogeneous points in the second image</param>
/// <param name="solutions">[Output] Possible solutions for the fundamental matrix</param>
void sevenPointMethod(const Mat& points1, const Mat& points2, vector<Mat>& solutions);

/// <summary>
/// Estimate the fundamental matrix between two images using eight point correspondences.
/// </summary>
/// <param name="points1">Nx3 matrix of homogeneous points in the first image</param>
/// <param name="points2">Nx3 matrix of homogeneous points in the second image</param>
/// <param name="solutions">[Output] Possible solutions for the fundamental matrix</param>
void eightPointMethod(const Mat& points1, const Mat& points2, vector<Mat>& solutions);

/// <summary>
/// Estimate the fundamental matrix robustly with RANSAC and the four point method.
/// </summary>
/// <param name="solver">Details which solver we should use to estimate the fundamental matrix</param>
/// <param name="points1">Nx3 matrix of homogeneous points in the first image</param>
/// <param name="points2">Nx3 matrix of homogeneous points in the second image</param>
/// <param name="iterations">Maximum number of iterations</param>
/// <param name="threshold">Error threshold for inlier/outlier calculation</param>
/// <returns>Estimated fundamental matrix</returns>
Mat estimateFundamentalMatrix(const CustomSolver solver, const Mat& points1, const Mat& points2, int iterations = 500, float threshold = 1.0);

/// <summary>
/// Count the number of inliers in a set of point correspondences based on a given
/// Fundamental matrix and a threshold for error in the epipolar constraint.
/// </summary>
/// <param name="F">Fundamental matrix</param>
/// <param name="points1">Nx3 matrix of homogeneous points in the first image</param>
/// <param name="points2">Nx3 matrix of homogeneous points in the second image</param>
/// <param name="threshold">Error threshold for inlier/outlier calculation</param>
/// <returns>Number of inliers</returns>
int countInliersFundamental(const Mat& F, const Mat& points1, const Mat& points2, float threshold);

/// <summary>
/// Estimate the fundamental matrix between two image pairs using SIFT descriptors
/// and the four point estimator.
/// </summary>
/// <param name="img1">First image</param>
/// <param name="img2">Second image</param>
/// <returns>Estimated fundamental matrix</returns>
Mat fundamentalFromImagePair(const Mat& img1, const Mat& img2);

/// <summary>
/// Decompose a Fundamental matrix into a relative rotation and translation that are
/// consistent with outward facing spherical motion.
/// </summary>
/// <param name="F">Fundamental matrix</param>
/// <param name="K">Camera intrinsic matrix</param>
/// <param name="R">[Output] Rotation matrix</param>
/// <param name="t">[Output] Translation vector</param>
/// <returns>Estimated fundamental matrix</returns>
void decomposeFundamentalMat(const Mat& F, const Mat& K, Mat& R, Mat& t);