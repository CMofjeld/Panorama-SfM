/*
* CSS 587 - Advance Computer Vision
* Spring 2021
* Carl Mofjeld, Drew Nelson
*
* Description:
* Methods to help render a 3D scene on the screen
* 
* NOTE: Source not in use as we were not able to get OpenCV's reconstruct methods
* to work
*/

#pragma once

#include <iostream>
#include <fstream>
#include <cstdint>
#include <filesystem>

#include <opencv2/sfm.hpp>
#include <opencv2/viz.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;
using namespace cv::sfm;

/// <summary>
/// Attempt to render 3D points in a window
/// </summary>
/// <param name="windowName">Name of the display window</param>
/// <param name="worldPoints">Points to render</param>
/// <returns>True if able to reconstruct the scene | False otherwise</returns>
bool renderScene(String windowName, vector<Vec3f> worldPoints);

/// <summary>
/// Attempt to generate a 3D scene from a video 
/// </summary>
/// <param name="videoPath">Full path to the video</param>
/// <param name="windowName">Name of the window to render the scene in</param>
void renderSceneFromVideo(String videoPath, String windowName);

void Experimenting();