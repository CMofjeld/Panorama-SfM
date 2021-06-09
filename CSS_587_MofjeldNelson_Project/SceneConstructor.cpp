/*
* CSS 587 - Advance Computer Vision
* Spring 2021
* Carl Mofjeld, Drew Nelson
*
* Description:
* Methods to help render a 3D scene on the screen
*
* NOTE: Source not in use as we were not able to get OpenCV's reconstruct methods
* to work. After different attempts and alterations, we decided to focus on the 
* accuracy of our fundamental matrix calculation
*/

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
#include "FundamentalTests.h"

#include "FundamentalSolvers.h"

using namespace std;
using namespace cv;
using namespace cv::sfm;

/// <summary>
/// Attempt to render 3D points in a window
/// </summary>
/// <param name="windowName">Name of the display window</param>
/// <param name="worldPoints">Points to render</param>
/// <returns>True if able to reconstruct the scene | False otherwise</returns>
bool renderScene(String windowName, vector<Mat> worldPoints)
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
		printf("Unable to render scene, no valid points provided\n");
	}

	return sceneRendered;
}

/// <summary>
/// We're not entirely sure if a intrinsic matrix needs to be provided to the reconstruct method (since we're seeing some errors). For now
/// let's provide an intrinsic matrix in the format of what we're eventually expecting while we investigate the reconstruct issue.
/// </summary>
/// <returns></returns>
Mat debug_GetTestInstrinic()
{
	float focalLength = 1200; //Not sure what the exact focal length of the galaxy s9+ is, but let's use the paper's default focal length to start
	Mat instrinsicMatrix(cv::Size(3, 3), CV_32F, Scalar(0));

	instrinsicMatrix.at<float>(0, 0) = focalLength;
	instrinsicMatrix.at<float>(1, 1) = focalLength;
	instrinsicMatrix.at<float>(2, 2) = 1;
	return instrinsicMatrix;
}

/// <summary>
/// As a last ditch effort for using OpenCV's reconstruct method, lets try using the reconstruct method that takes in an array of 
/// images. Convert a video stream to images on the disk, and then call the reconstruct method.
/// </summary>
/// <param name="videoPath">Full path to the video</param>
/// <param name="windowName">Name of the window to render the scene in</param>
void renderSceneFromVideo(String videoPath, String windowName)
{
	vector<String> videoFramePaths = saveVideoToImages(videoPath);
	if (videoFramePaths.size() > 0)
	{
		vector<Mat> points3d_estimated;
		//vector<Vec3f> points3d_estimated, K;
		Mat intrinsicMatrix = debug_GetTestInstrinic();
		vector<Mat> Ps;
		cv::sfm::reconstruct(videoFramePaths, Ps, points3d_estimated, intrinsicMatrix, true);

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

/// <summary>
/// Structures the set of points into the track/frame relationship as shown here: 
/// https://docs.opencv.org/master/d5/dab/tutorial_sfm_trajectory_estimation.html. Even though this example
/// is strictly for camera pose estimation, we'll apply the same input format for our reconstruction attempts
/// </summary>
/// <param name="points">Lists of 2D points to format</param>
/// <returns>
/// Input array of arrays which contains eseentially a vector of frames in which each frame contains all the 
/// x, y positions of all the points for a given frame
/// </returns>
vector<Mat> formatToInputArrayOfArrays(vector<vector<Point2f>> points)
{
	vector<Mat> points2d;

	int numOfFrames = points.size();
	int numOfTracks = points[0].size();

	for (int i = 0; i < numOfFrames; i++)
	{
		Mat_<double> frame(2, numOfTracks);
		for (int j = 0; j < numOfTracks; j++)
		{
			Point2f point = points[i][j];
			frame(0, j) = point.x; // abs(point.x); //Seeing negative points here, when I'm not sure we're supposed to? 
			frame(1, j) = point.y; // abs(point.y);
		}
		points2d.push_back(Mat(frame));
	}
	return points2d;
}



/// <summary>
/// A test method to perform quick experiments as we try to figure out issues with the reconstruction
/// </summary>
void Experimenting()
{
	VideoCapture loadedVideo;

	if (loadVideo("short_test_vid.mp4", loadedVideo))
	{
		int frameCount = 0;
		vector<Mat> captures;
		while (loadedVideo.isOpened())
		{
			Mat frame;
			if (loadedVideo.read(frame))
			{
				// Maybe we need frames that are more different from one another? Let's try jumping frames so that the frames are hopefully more distinct
				if (frameCount % 10 == 0)
				{
					captures.push_back(frame);
				}
				frameCount++;
			}
			else
			{
				break;
			}
		}

		// Ignore the calculated fundamental matrix for now, simply get the point correspondences between two images
		vector<Point2f> matches1, matches2;
		Mat fundamentalMatrix = fundamentalFromImagePair(captures[0], captures[1], matches1, matches2);

		vector<vector<Point2f>> unformattedPoints;
		unformattedPoints.push_back(matches1);
		unformattedPoints.push_back(matches2);

		vector<Mat> points = formatToInputArrayOfArrays(unformattedPoints);
		vector<Mat> Rs_est, ts_est, Ps, points3d_estimated;
		Mat intrinsicMatrix = debug_GetTestInstrinic();
		reconstruct(points, Ps, points3d_estimated, intrinsicMatrix, true);
		//reconstruct(points, Rs_est, ts_est, intrinsicMatrix, points3d_estimated, true);
		
		if (points3d_estimated.size() > 0)
		{
			cout << "Points found!" << endl;
		}
		else
		{
			cout << "No points recognized" << endl;
		}
	}
}

/// <summary>
/// Another experimentation method in which we try to get the 3D points using the sfm::triangulatePoints method
/// We unfortunately didn't get too far with this attempt and decided to press on with focus on the fundamental matrix
/// calculation
/// </summary>
void Experimenting_Triangulation()
{
	Mat fundamentalMatrix;
	vector<Point2f> points1, points2;
	SfmTesting(fundamentalMatrix, points1, points2);

	Mat projectionMat1, projectionMat2;
	projectionsFromFundamental(fundamentalMatrix, projectionMat1, projectionMat2);

	Mat intrinsicMatrix = debug_GetTestInstrinic();
	vector<Mat> ps;

	vector<vector<Point2f>> initialPointFormat;
	initialPointFormat.push_back(points1);
	initialPointFormat.push_back(points2);

	vector<Mat> points2d = formatToInputArrayOfArrays(initialPointFormat);
	vector<Mat> projection_matricies;


	projection_matricies.push_back(projectionMat1);
	projection_matricies.push_back(projectionMat2);

	vector<Mat> Rs_est, ts_est, points3d_estimated;
	vector<Mat> Ps;

	triangulatePoints(points2d, projection_matricies, points3d_estimated);

	if (renderScene("Testing", points3d_estimated))
	{
		printf("We got something!");
	}
	else
	{
		printf("Error occurred\n");
	}
}