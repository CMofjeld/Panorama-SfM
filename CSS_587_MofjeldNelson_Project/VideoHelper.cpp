/*
* CSS 587 - Advance Computer Vision
* Spring 2021
* Carl Mofjeld, Drew Nelson
*
* Description:
* TODO: Add more description for the file when we're further along in the project
*/

#include "VideoHelper.h"

struct stat info;

/// <summary>
/// Read a video from the local "Input" directory into memory.
/// </summary>
/// <param name="fileName">Full file path to the video to load</param>
/// <param name="cap">[Output] Loaded video file</param>
/// <returns>True if video loaded succesfully. False otherwise</returns>
bool loadVideo(String fileName, VideoCapture& cap)
{
	cap = VideoCapture("Input\\" + fileName);
	bool videoLoaded = cap.isOpened();
	if (!videoLoaded)
	{
		printf("Failed to read video at %s", fileName.c_str());
	}
	return videoLoaded;
}

/// <summary>
/// Attempt to delete a directory and its contents. 
/// </summary>
/// <param name="directoryPath">Full path of the directory to remove</param>
/// <returns>True if directory deleted. False otherwise</returns>
bool deleteDirectory(String directoryPath)
{
	bool directoryRemoved = true;

	// Try deleting the directory if it exists
	if (stat(directoryPath.c_str(), &info) == 0)
	{
		error_code errorCode;
		if (!filesystem::remove_all(directoryPath, errorCode))
		{
			printf("Error occurred deleting video dump directory: %s\n", errorCode.message().c_str());
			directoryRemoved = false;
		}
	}

	// Confirm that the directory doesn't exists
	directoryRemoved = stat(directoryPath.c_str(), &info) != 0;

	return directoryRemoved;
}

/// <summary>
/// Attempt to create a directory (if it doesn't exist already)
/// </summary>
/// <param name="directoryPath">Path of the directory to create</param>
/// <returns>True if directory exists | False otherwise</returns>
bool createDirectory(String directoryPath)
{
	bool directoryCreated = true;

	// Attempt to create the directory
	if (stat(directoryPath.c_str(), &info) != 0)
	{
		error_code errorCode;
		if (!filesystem::create_directory(directoryPath, errorCode))
		{
			printf("Error occurred creating video dump directory: %s", errorCode.message().c_str());
			directoryCreated = false;
		}
	}

	// Check to see if the directory was actually made
	directoryCreated = stat(directoryPath.c_str(), &info) == 0;

	return directoryCreated;
}

/// <summary>
/// Save each frame in a video to a file local on the disk
/// </summary>
/// <param name="videoPath">Full path to the video</param>
/// <returns>List of file paths to each video frame saved to the disk</returns>
vector<String> saveVideoToImages(String videoPath)
{
	VideoCapture loadedVideo;

	if (loadVideo(videoPath, loadedVideo))
	{
		String videoDumpDir = "video_dump"; // We'll just try to save the files in a local dir

		// Clear our our temp directory if it exists and try to recreate it
		if (deleteDirectory(videoDumpDir) && createDirectory(videoDumpDir))
		{
			int frameCount = 0;
			vector<String> captures;
			while (loadedVideo.isOpened())
			{
				Mat frame;
				if (loadedVideo.read(frame))
				{
					String filePath = videoDumpDir + "\\" + to_string(frameCount++) + "_cap.jpg";
					captures.push_back(filePath);
					imwrite(filePath, frame);
				}
				else
				{
					break;
				}
			}
			return captures;
		}
		return {};
	}
}