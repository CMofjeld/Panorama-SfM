# Panorama SfM
#### Authors: Carl Mofjeld, Drew Nelson

This project attempts to follow the methods and recreate the results the "Structure From Motion for Panorama-Style Videos" research done by the University of Washington and Facebook Reality Labs [as seen here](https://arxiv.org/abs/1906.03539). 

This work was done as part of the final project in the CSS 587 - Advance Topics Computer Vision course and the Univerity Washington, Bothell during the spring quarter 2021.

# Dependencies and Setup
This project relies on the the SfM and Viz modules from OpenCV. As a result, OpenCV the OpenCV contribution models we're build using CMAKE targeting VS 2019. Additionally, the SfM module, required the libraries Eigen, GLog, GFlags, and Ceres Solver. As a result, these libraries were built using CMAKE as well. 

Unfortunately, due to git's size restriction, we were unable to provide all of the dependencies in the repository. To build and run the project, you must download the [dependencies files](https://drive.google.com/file/d/1FtKI8EJpFOQ8zwwsl6AM2_yBAWfsF1n-/view?usp=sharing) and unpackage them into the CSS_587_MofjeldNelson_Project directory. After unpacking the file, you should have the following folders under your CSS_587_MofjeldNelson_Project folder:
* bin
* includes
* Input
* libs
 
Lastly, you'll need to add the full path of the project's bin directory (i.e. D:\UWB\CSS_587\Panorama-SfM\CSS_587_MofjeldNelson_Project\bin) to your PC's PATH variable.