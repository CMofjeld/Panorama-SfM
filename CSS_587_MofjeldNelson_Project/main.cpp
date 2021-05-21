/*
* CSS 587 - Advance Computer Vision
* Spring 2021
* Carl Moffjeld, Drew Nelson
* 
* Description:
* TODO: Add more description for the file when we're further along in the project
*/

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <chrono>
#include <ctime>  
#include <glog/logging.h>
#include <Eigen/Dense>
#include "gflags/gflags.h"
#include <math.h>
#include "ceres/ceres.h"
#include <opencv2/viz.hpp>
#include <opencv2/sfm.hpp>

using namespace cv;
using namespace std;

using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solve;
using ceres::Solver;
// A templated cost functor that implements the residual r = 10 -
// x. The method operator() is templated so that we can then use an
// automatic differentiation wrapper around it to generate its
// derivatives.
struct CostFunctor {
    template <typename T>
    bool operator()(const T* const x, T* residual) const {
        residual[0] = 10.0 - x[0];
        return true;
    }
};

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
/// A quick method to run a method from glog to confirm we're able to
/// run methods from this dependency
/// </summary>
void TestGlog()
{
    // Simple test to ensure we have glog configured correctly
    google::InitGoogleLogging("");
    LOG(INFO) << "GLOG configured correctly";
    printf("Glog configured correctly\n");
}

/// <summary>
/// A quick method to run a method from eigen to confirm we're able to
/// run methods from this dependency
/// </summary>
void TestEigen()
{
    Eigen::MatrixXd m(2, 2);
    m(0, 0) = 3;
    m(1, 0) = 2.5;
    m(0, 1) = -1;
    m(1, 1) = m(1, 0) + m(0, 1);
    printf("Eigen configured correctly\n");
}

/// <summary>
/// A quick method to run a method from gflags to confirm we're able to
/// run methods from this dependency
/// </summary>
void TestGFlags()
{
    gflags::SetVersionString("1.0.0");
    printf("GFlags configured correctly\n");
}

void TestCeres()
{
    // The variable to solve for with its initial value. It will be
    // mutated in place by the solver.
    double x = 0.5;
    const double initial_x = x;
    // Build the problem.
    Problem problem;
    // Set up the only cost function (also known as residual). This uses
    // auto-differentiation to obtain the derivative (jacobian).
    CostFunction* cost_function =
        new AutoDiffCostFunction<CostFunctor, 1, 1>(new CostFunctor);
    problem.AddResidualBlock(cost_function, nullptr, &x);
    // Run the solver!
    Solver::Options options;
    options.minimizer_progress_to_stdout = true;
    Solver::Summary summary;
    Solve(options, &problem, &summary);

    printf("Ceres configured correctly\n");
}

/// <summary>
/// Run a few methods to ensure our dependencies have all loaded
/// </summary>
void TestDependencies()
{
    TestGlog();
    TestEigen();
    TestGFlags();
    TestCeres();
}

/// <summary>
/// Application entry point. 
/// TODO: Add further description when we're further along in the project
/// </summary>
/// <returns></returns>
int main(int argc, char* argv[])
{
    //TestDependencies();

    //VideoCapture loadedVideo;
    int exitValue = -1;
    //if (LoadVideo("short_test_vid.mp4", loadedVideo)) 
    //{
    //    double fps = loadedVideo.get(CAP_PROP_FPS);
    //    String windowName = "Test Video";

    //    namedWindow(windowName, WINDOW_AUTOSIZE);
    //    while (loadedVideo.isOpened())
    //    {
    //        Mat frame;
    //        // Check to see if we've reached the end or if the user has hit the escape key
    //        if (!loadedVideo.read(frame) || waitKey(30) == 27)
    //        {
    //            break;
    //        }
    //        imshow(windowName, frame);
    //    }
    //    destroyWindow(windowName);
    //    loadedVideo.release();
    //    exitValue = 0;
    //}

    viz::Viz3d window("Coordinate Frame");
    window.setWindowSize(Size(500, 500));
    window.setWindowPosition(Point(150, 150));
    window.setBackgroundColor(); // black by default

    vector<Vec3f> point_cloud_est;

    point_cloud_est.push_back(Vec3f(0, 0, 10));
    point_cloud_est.push_back(Vec3f(0, 10, 0));
    point_cloud_est.push_back(Vec3f(10, 0, 0));

    if (point_cloud_est.size() > 0)
    {
        cout << "Rendering points   ... ";
        viz::WCloud cloud_widget(point_cloud_est, viz::Color::green());
        window.showWidget("point_cloud", cloud_widget);
        cout << "[DONE]" << endl;
    }
    else
    {
        cout << "Cannot render points: Empty pointcloud" << endl;
    }

    window.spin();

    printf("Program finished");
    waitKey(0);
    return exitValue;
}