/*
* CSS 587 - Advance Computer Vision
* Spring 2021
* Carl Mofjeld, Drew Nelson
* 
* Description:
* Entry method for the main application. Starts the trials for testing the
* different solvers.
* 
* To run the application, the application assumes that the configuration steps as mentioned
* on the project's github page have been followed: https://github.com/CMofjeld/Panorama-SfM
*/

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

#include "FundamentalSolvers.h"
#include "FundamentalTests.h"
#include "SceneConstructor.h"

using namespace cv;
using namespace std;

/// <summary>
/// Application entry point. Starts the application test trials - calculate a fundmental matrix using each solver, then compares the results
/// against synthetic data
/// </summary>
void main(int argc, char* argv[])
{
    const int NUM_ZERO_NOISE_TRIAL = 1000;
    const int NUM_TRIALS_PER_NOISE_LEVEL = 500;

    runAllTrials(NUM_ZERO_NOISE_TRIAL, NUM_TRIALS_PER_NOISE_LEVEL);
}