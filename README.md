# Panorama SfM
#### Authors: Carl Mofjeld, Drew Nelson

This project attempts to follow the methods and recreate the results the "Structure From Motion for Panorama-Style Videos" research done by the University of Washington and Facebook Reality Labs [as seen here](https://arxiv.org/abs/1906.03539). 

This work was done as part of the final project in the CSS 587 - Advance Topics Computer Vision course and the Univerity Washington, Bothell during the spring quarter 2021.

# Dependencies and Setup
This project relies on the the SfM and Viz modules from OpenCV. As a result, OpenCV the OpenCV contribution models we're build using CMAKE targeting VS 2019. Additionally, the SfM module, required the libraries Eigen, GLog, GFlags, and Ceres Solver. As a result, these libraries were built using CMAKE as well. 

All of the dependencies needed for this project are already included with the project. The only step required is to add the project's bin to your PC's PATH environment variable (e.g. D:\UWB\CSS_587\Panorama-SfM\CSS_587_MofjeldNelson_Project\bin).