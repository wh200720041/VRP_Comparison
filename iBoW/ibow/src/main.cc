/**
* This file is part of ibow-lcd.
*
* Copyright (C) 2017 Emilio Garcia-Fidalgo <emilio.garcia@uib.es> (University of the Balearic Islands)
*
* ibow-lcd is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ibow-lcd is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ibow-lcd. If not, see <http://www.gnu.org/licenses/>.
*/

#include <iostream>

#include <opencv2/features2d.hpp>
#include "Util.h"
#include "lcdetector.h"
/*
auto saliency_timer_start = std::chrono::high_resolution_clock::now();
salienceFilter.saliency_extraction(image, saliency_map);
auto saliency_timer_end = std::chrono::high_resolution_clock::now();
std::cout << "saliency time=" << double(std::chrono::duration_cast<std::chrono::milliseconds>(saliency_timer_end - saliency_timer_start).count()) << std::endl;
fout << double(std::chrono::duration_cast<std::chrono::milliseconds>(saliency_timer_end - saliency_timer_start).count()) << "\n";
*/
int main(int argc, char** argv) {
  // Creating feature detector and descriptor
  cv::Ptr<cv::Feature2D> detector = cv::ORB::create(1000);  // Default params

  // Loading image filenames

  cv::VideoCapture video("../../../Dataset/KITTI/sequence00//KITTI1_grey.avi");
  int width = (int)video.get(cv::CAP_PROP_FRAME_WIDTH);
  int height = (int)video.get(cv::CAP_PROP_FRAME_HEIGHT);
  int video_size = (int)video.get(cv::CAP_PROP_FRAME_COUNT);
  int frame_rate = (int)video.get(cv::CAP_PROP_FPS);
  std::ofstream fout("ibow_all_backup.txt");

  // Creating the loop closure detector object
  ibow_lcd::LCDetectorParams params;  // Assign desired parameters
  ibow_lcd::LCDetector lcdet(params);
  std::cout << "timer start ...." << std::endl;
  auto t_start = std::chrono::high_resolution_clock::now();
  // Processing the sequence of images
  for (unsigned i = 0; i < video_size; i++) {
    // Processing image i
    std::cout << "--- Processing image " << i << std::endl;
		
    // Loading and describing the image
	cv::Mat img = read_image(video, i);
	auto ibow_timer_start = std::chrono::high_resolution_clock::now();
    std::vector<cv::KeyPoint> kps;
    detector->detect(img, kps);
    cv::Mat dscs;
    detector->compute(img, kps, dscs);

    ibow_lcd::LCDetectorResult result;
    lcdet.process(i, kps, dscs, &result);


    switch (result.status) {
      case ibow_lcd::LC_DETECTED:
        std::cout << "--- Loop detected!!!: " << result.train_id <<
                     " with " << result.inliers << " inliers" << std::endl;
        break;
      case ibow_lcd::LC_NOT_DETECTED:
        std::cout << "No loop found" << std::endl;
        break;
      case ibow_lcd::LC_NOT_ENOUGH_IMAGES:
        std::cout << "Not enough images to found a loop" << std::endl;
        break;
      case ibow_lcd::LC_NOT_ENOUGH_ISLANDS:
        std::cout << "Not enough islands to found a loop" << std::endl;
        break;
      case ibow_lcd::LC_NOT_ENOUGH_INLIERS:
        std::cout << "Not enough inliers" << std::endl;
        break;
      case ibow_lcd::LC_TRANSITION:
        std::cout << "Transitional loop closure" << std::endl;
        break;
      default:
        std::cout << "No status information" << std::endl;
        break;
    }
	auto ibow_timer_end = std::chrono::high_resolution_clock::now();
	std::cout << "ibow time=" << double(std::chrono::duration_cast<std::chrono::milliseconds>(ibow_timer_end - ibow_timer_start).count()) << std::endl;
	fout << double(std::chrono::duration_cast<std::chrono::milliseconds>(ibow_timer_end - ibow_timer_start).count()) << "\n";
  }
  auto t_end = std::chrono::high_resolution_clock::now();
  std::cout << "dbow time=" << double(std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count()) / 1000.0 << " s" << std::endl;
  fout.close();
  std::cout << "end" << std::endl;
  return 0;
}
