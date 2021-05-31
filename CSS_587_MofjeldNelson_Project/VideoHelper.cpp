/*
* CSS 587 - Advance Computer Vision
* Spring 2021
* Carl Mofjeld, Drew Nelson
*
* Description:
* Utility methods to use when interacting with the input video
*/

#include "VideoHelper.h"

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