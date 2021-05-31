/*
* CSS 587 - Advance Computer Vision
* Spring 2021
* Carl Mofjeld, Drew Nelson
*
* Description:
* Utility methods to use when interacting with the input video
*/

#pragma once

#include <filesystem>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include "FileHelper.h"

using namespace std;
using namespace cv;

/// <summary>
/// Read a video from the local "Input" directory into memory.
/// </summary>
/// <param name="fileName">Full file path to the video to load</param>
/// <param name="cap">[Output] Loaded video file</param>
/// <returns>True if video loaded succesfully. False otherwise</returns>
bool loadVideo(String fileName, VideoCapture& cap);

/// <summary>
/// Save each frame in a video to a file local on the disk
/// </summary>
/// <param name="videoPath">Full path to the video</param>
/// <returns>List of file paths to each video frame saved to the disk</returns>
vector<String> saveVideoToImages(String videoPath);