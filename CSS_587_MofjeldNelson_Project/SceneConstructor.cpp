#define CERES_FOUND 1

#include <opencv2/sfm.hpp>
#include <opencv2/viz.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <iostream>

#include <iostream>
#include <fstream>
#include <cstdint>
#include <filesystem>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include <direct.h>

#include "VideoHelper.h"

using namespace std;
using namespace cv;
using namespace cv::sfm;

/// <summary>
/// Attempt to render 3D points in a window
/// </summary>
/// <param name="windowName">Name of the display window</param>
/// <param name="worldPoints">Points to render</param>
/// <returns>True if able to reconstruct the scene | False otherwise</returns>
bool renderScene(String windowName, vector<Vec3f> worldPoints)
{
	bool sceneRendered = false;

	if (worldPoints.size() > 0)
	{
		viz::Viz3d window(windowName);
		window.setWindowSize(Size(500, 500));
		window.setWindowPosition(Point(100, 100));
		window.setBackgroundColor(); 

		viz::WCloud cloud_widget(worldPoints, viz::Color::green());
		window.showWidget("point_cloud", cloud_widget);
		window.spin();
		sceneRendered = true;
	}
	else
	{
		printf("Unable to render scene, no valid points provided");
	}

	return sceneRendered;
}

Mat debug_GetTestInstrinic()
{
	float s9GalaxyFocalLength = 26;
	Mat instrinsicMatrix(cv::Size(3, 3), CV_32F, Scalar(0));

	instrinsicMatrix.at<float>(0, 0) = s9GalaxyFocalLength;
	instrinsicMatrix.at<float>(1, 1) = s9GalaxyFocalLength;
	instrinsicMatrix.at<float>(2, 2) = 1;
	return instrinsicMatrix;
}

/// <summary>
/// Attempt to generate a 3D scene from a video 
/// </summary>
/// <param name="videoPath">Full path to the video</param>
/// <param name="windowName">Name of the window to render the scene in</param>
void renderSceneFromVideo(String videoPath, String windowName)
{
	vector<String> videoFramePaths = saveVideoToImages(videoPath);
	if (videoFramePaths.size() > 0)
	{
		vector<Vec3f> points3d_estimated, K;
		Mat intrinsicMatrix = debug_GetTestInstrinic();
		Mat ps;
		cv::sfm::reconstruct(videoFramePaths, ps, points3d_estimated, intrinsicMatrix, true);

		if (points3d_estimated.size() > 0)
		{
			viz::Viz3d window(windowName);
			window.setWindowSize(Size(500, 500));
			window.setWindowPosition(Point(150, 150));
			window.setBackgroundColor(); // black by default

			viz::WCloud cloud_widget(points3d_estimated, viz::Color::green());
			window.showWidget("point_cloud", cloud_widget);

			window.spin();
		}
	}
}