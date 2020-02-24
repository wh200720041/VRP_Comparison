#pragma once

#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm> 
// DBoW3

#include <DBoW3.h>
#include <DescManip.h>

// OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#ifdef USE_CONTRIB
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/xfeatures2d.hpp>
#endif


using namespace DBoW3;
using namespace std;
using namespace cv;


#define DBOW_THRESHOLD 0.06

//vocal setting
const int k = 10;
const int L = 6;
const WeightingType weight = TF_IDF;
const ScoringType score = L1_NORM;

const int FEATURE_NUMBER = 500;

void build_vocabulary(vector<string> path_to_images);
//build and save small vol
void build_vocabulary(cv::VideoCapture &video);

cv::Mat read_image(cv::VideoCapture &video, int number);

void get_keypoint_and_descriptor(cv::Mat image, vector<KeyPoint>& keypoint, cv::Mat& descriptor);

void display_matched_image(cv::Mat image1, vector<KeyPoint> keypoint1, cv::Mat image2, vector<KeyPoint> keypoint2, std::vector< DMatch > matches);