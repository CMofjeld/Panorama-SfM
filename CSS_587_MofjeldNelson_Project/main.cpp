/*
* CSS 587 - Advance Computer Vision
* Spring 2021
* Carl Mofjeld, Drew Nelson
* 
* Description:
* Entry method for the main application. Starts the trials for testing the
* different solvers
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
/// <returns></returns>
int main(int argc, char* argv[])
{
    const int NUM_ZERO_NOISE_TRIAL = 10000;
    const int NUM_TRIALS_PER_NOISE_LEVEL = 1000;

    runAllTrials(NUM_ZERO_NOISE_TRIAL, NUM_TRIALS_PER_NOISE_LEVEL);

    return 0;
}