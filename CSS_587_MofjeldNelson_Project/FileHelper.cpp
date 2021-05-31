/*
* CSS 587 - Advance Computer Vision
* Spring 2021
* Carl Mofjeld, Drew Nelson
*
* Description:
* Utility methods interacting with files
*/

#include "FileHelper.h"

/// <summary>
/// Attempt to delete a directory and its contents. 
/// </summary>
/// <param name="directoryPath">Full path of the directory to remove</param>
/// <returns>True if directory deleted. False otherwise</returns>
bool deleteDirectory(string directoryPath)
{
	bool directoryRemoved = true;
	struct stat info;

	// Try deleting the directory if it exists
	if (stat(directoryPath.c_str(), &info) == 0)
	{
		error_code errorCode;
		if (!filesystem::remove_all(directoryPath, errorCode))
		{
			printf("Error occurred deleting video dump directory: %s\n", errorCode.message().c_str());
			directoryRemoved = false;
		}
	}

	// Confirm that the directory doesn't exists
	directoryRemoved = stat(directoryPath.c_str(), &info) != 0;

	return directoryRemoved;
}

/// <summary>
/// Attempt to delete a file if the file exists
/// </summary>
/// <param name="filePath">Path to the file to delete</param>
/// <returns>True if file deleted | False otherwise</returns>
bool deleteFile(string filePath)
{
	struct stat fileInfo;
	// Delete the previously existing file (if one exists)
	if (stat(filePath.c_str(), &fileInfo) == 0)
	{
		if (remove(filePath.c_str()) != 0)
		{
			printf("Error occurred deleting existing file\n");
			return false;
		}
	}
	return true;
}

/// <summary>
/// Attempt to create a directory (if it doesn't exist already)
/// </summary>
/// <param name="directoryPath">Path of the directory to create</param>
/// <returns>True if directory exists | False otherwise</returns>
bool createDirectory(string directoryPath)
{
	bool directoryCreated = true;
	struct stat info;

	// Attempt to create the directory
	if (stat(directoryPath.c_str(), &info) != 0)
	{
		error_code errorCode;
		if (!filesystem::create_directory(directoryPath, errorCode))
		{
			printf("Error occurred creating video dump directory: %s", errorCode.message().c_str());
			directoryCreated = false;
		}
	}

	// Check to see if the directory was actually made
	directoryCreated = stat(directoryPath.c_str(), &info) == 0;

	return directoryCreated;
}