/*
* CSS 587 - Advance Computer Vision
* Spring 2021
* Carl Mofjeld, Drew Nelson
* 
* Description:
* TODO: Add more description for the file when we're further along in the project
*/

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

#include "DependencyChecker.h"
#include "FundamentalSolvers.h"

using namespace cv;
using namespace std;

/// <summary>
/// Read a video from the local "Input" directory into memory.
/// </summary>
/// <param name="fileName">Full file path to the video to load</param>
/// <param name="cap">[Output] Loaded video file</param>
/// <returns>True if video loaded succesfully. False otherwise</returns>
bool LoadVideo(String fileName, VideoCapture &cap)
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
/// Application entry point. 
/// TODO: Add further description when we're further along in the project
/// </summary>
/// <returns></returns>
int main(int argc, char* argv[])
{
    if (!testDependencies())
    {
        return -1;
    }

    int exitValue = 0;
    VideoCapture loadedVideo;
    
    if (LoadVideo("short_test_vid.mp4", loadedVideo)) 
    {
        String windowName = "Test Video";

        namedWindow(windowName, WINDOW_AUTOSIZE);
        while (loadedVideo.isOpened())
        {
            Mat frame;
            // Check to see if we've reached the end or if the user has hit the escape key
            if (!loadedVideo.read(frame) || waitKey(30) == 27)
            {
                break;
            }
            imshow(windowName, frame);
        }
        destroyWindow(windowName);
        loadedVideo.release();
    }
    else
    {
        exitValue = -1;
    }

    cout << "Testing four point method:" << endl;
    testFourPoint();
    cout << endl << "Testing RANSAC:" << endl;
    testEstimateFundamentalMatrix();

    printf("Program finished");
    waitKey(0);
    return exitValue;
}