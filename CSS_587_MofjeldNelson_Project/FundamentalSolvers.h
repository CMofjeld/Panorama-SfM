#pragma once
#include <opencv2/core.hpp>
#include <vector>

using namespace cv;
using namespace std;

/// <summary>
/// Estimate the fundamental matrix between two images using four point correspondences.
/// </summary>
/// <param name="points1">List of four points in the first image</param>
/// <param name="points2">List of four points in the second image</param>
/// <returns>Estimated fundamental matrix</returns>
Mat fourPointMethod(const vector<Point2f>& points1, const vector<Point2f>& points2);

/// <summary>
/// Evaluate the accuracy of fourPointMethod().
/// </summary>
void testFourPoint();