/*
* CSS 587 - Advance Computer Vision
* Spring 2021
* Carl Mofjeld, Drew Nelson
*
* Description:
* Utility methods interacting with files
*/

#pragma once

#include <filesystem>

using namespace std;

/// <summary>
/// Attempt to delete a directory and its contents. 
/// </summary>
/// <param name="directoryPath">Full path of the directory to remove</param>
/// <returns>True if directory deleted. False otherwise</returns>
bool deleteDirectory(string directoryPath);

/// <summary>
/// Attempt to delete a file if the file exists
/// </summary>
/// <param name="filePath">Path to the file to delete</param>
/// <returns>True if file deleted | False otherwise</returns>
bool deleteFile(string filePath);

/// <summary>
/// Attempt to create a directory (if it doesn't exist already)
/// </summary>
/// <param name="directoryPath">Path of the directory to create</param>
/// <returns>True if directory exists | False otherwise</returns>
bool createDirectory(string directoryPath);