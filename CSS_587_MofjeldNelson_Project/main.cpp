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
#include "SceneConstructor.h"

using namespace cv;
using namespace std;

/// <summary>
/// Application entry point. 
/// TODO: Add further description when we're further along in the project
/// </summary>
/// <returns></returns>
int main(int argc, char* argv[])
{
    int exitValue = 0;

    //renderSceneFromVideo("short_test_vid.mp4", "OpenCV Construction");
    vector<Vec3f> testPoints;

    testPoints.push_back(Vec3f(10, 0, 0));
    testPoints.push_back(Vec3f(0, 10, 0));
    testPoints.push_back(Vec3f(0, 0, 10));

    renderScene("Test Scene", testPoints);

    waitKey(0);
    return exitValue;
}