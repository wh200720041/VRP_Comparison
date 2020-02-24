#pragma once
#ifndef  DESCRIPTOR_FILTER
#define DESCRIPTOR_FILTER

#include "Util.h"

cv::Mat extractDiscriptor(vector<Mat> &descriptor, vector<vector<KeyPoint>> &keypoint);

cv::Mat extractDiscriptorWithFixedSize(vector<Mat> &descriptor, vector<vector<KeyPoint>> &keypoint, int size = 200);

cv::Mat extractDiscriptorWithMinSize(vector<Mat> &descriptor, vector<vector<KeyPoint>> &keypoint, int min_size = 100);

cv::Mat extractDiscriptorWithExtra(vector<Mat> &descriptor, vector<vector<KeyPoint>> &keypoint);

const int feature_size = 4000;


#endif // ! DESCRIPTORFILTER
