//  Copyright (c) 2017 Universitat de les Illes Balears
//  This file is part of LIBHALOC.
//
//  LIBHALOC is free software: you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation, either version 3 of the License, or
//  (at your option) any later version.
//
//  LIBHALOC is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//
//  You should have received a copy of the GNU General Public License
//  along with LIBHALOC. If not, see <http://www.gnu.org/licenses/>.


#include <fstream>
#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm> 

#include <opencv2/opencv.hpp>
#include "Util.h"
#include "hash.h"


/**
 * @brief      Main entry point
 *
 * @param[in]  argc  The argc
 * @param      argv  The argv
 *
 * @return     0
 */

int main(int argc, char** argv) {


	// Init feature extractor
	cv::Ptr<cv::Feature2D> feat(new cv::Feature2D());
	feat = cv::KAZE::create();

	// Hash object
	haloc::Hash haloc;

	// Set params
	haloc::Hash::Params params;
	params.max_desc = 100;
	haloc.SetParams(params);

	// Operational variables
	//int img_idx = 0;
	int discard_window = 10;
	std::map<int, std::vector<float> > hash_table;

	cv::VideoCapture video("../../../Dataset/KITTI/sequence00//KITTI1_grey.avi");
	int width = (int)video.get(cv::CAP_PROP_FRAME_WIDTH);
	int height = (int)video.get(cv::CAP_PROP_FRAME_HEIGHT);
	int video_size = (int)video.get(cv::CAP_PROP_FRAME_COUNT);
	int frame_rate = (int)video.get(cv::CAP_PROP_FPS);

	std::ofstream fout("haloc_feature_extraction.txt");
	auto t_start = std::chrono::high_resolution_clock::now();
	
	// Loop over the images
	for (int num_count = 0; num_count < video_size; num_count++) {

		cv::Mat image_read = read_image(video, num_count);
		std::cout << num_count << std::endl;
		cv::Mat image;
		cv::cvtColor(image_read, image, CV_BGR2GRAY);
		std::cout << "timer start ...." << std::endl;
		auto feature_extraction_start = std::chrono::high_resolution_clock::now();
		// Extract keypoints and descriptors
		cv::Mat desc;
		std::vector<cv::KeyPoint> kp;
		feat->detectAndCompute(image, cv::noArray(), kp, desc);

		// Compute the hash
		std::vector<float> hash = haloc.GetHash(kp, desc, image.size());
		hash_table.insert(std::pair<int, std::vector<float> >(num_count, hash));
		auto feature_extraction_end = std::chrono::high_resolution_clock::now();
		fout << double(std::chrono::duration_cast<std::chrono::milliseconds>(feature_extraction_end - feature_extraction_start).count())<< "\n";
	}
	fout.close();
	//double average_time = double(std::chrono::duration_cast<std::chrono::milliseconds>(feature_extraction_end - feature_extraction_start).count()) / video_size;
	// Find loop closings
	int num_temp = 0;
	
	//for (double eps = 0.4;eps < 0.51;eps = eps + 0.02){
	double eps = 0.4;
		//std::ostringstream stringStream;
		//stringStream << "result" << num_temp++<<".txt";
		//std::string copyOfStr = stringStream.str();
		//std::ofstream fout("haloc_feature_extraction.txt");

		for (uint i = 0; i < hash_table.size(); ++i) {
			std::cout << i << std::endl;
			auto haloc_start = std::chrono::high_resolution_clock::now();
			for (uint j = 0; j < i; ++j) {
				int dist_original = 0;
				int dist = 0;
				int neighbourhood = abs(i - j);
				if (neighbourhood > 500) {
					dist_original = haloc.CalcDist(hash_table[i], hash_table[j], eps);

					if (dist_original < 4) {
						dist = 0;
					}
					else {
						dist = 1;
						//fout << "t1=" << i << '\t' << "coincides with t2=" << j << "\n";
					}
				}


			}
			auto haloc_end = std::chrono::high_resolution_clock::now();
			//fout << double(std::chrono::duration_cast<std::chrono::milliseconds>(haloc_end - haloc_start).count())+ average_time << "\n";
		}
		fout.close();
	//}

	

	auto t_end = std::chrono::high_resolution_clock::now();
	std::cout << "dbow time=" << double(std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count()) / 1000.0 << " s" << std::endl;

	std::cout << "end" << std::endl;
  
  
  return 0;
}
