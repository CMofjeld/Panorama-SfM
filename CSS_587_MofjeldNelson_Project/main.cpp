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
#include "FundamentalTests.h"
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

    cout << "Testing four point method:" << endl;
    testFourPoint();
    cout << endl << "Testing RANSAC:" << endl;
    testEstimateFundamentalMatrix();

    //renderSceneFromVideo("short_test_vid.mp4", "OpenCV Construction");
    cout << "Testing point rendering" << endl;
    vector<Vec3f> testPoints;
    testPoints.push_back(Vec3f(10, 0, 0));
    testPoints.push_back(Vec3f(0, 10, 0));
    testPoints.push_back(Vec3f(0, 0, 10));
    renderScene("Test Scene", testPoints);

    printf("Program finished");

    waitKey(0);
    return exitValue;
}