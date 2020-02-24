#include <iostream>

#define SRRG_HBST_HAS_OPENCV

#include "binary_tree.hpp"
#include "Util.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm> 
// OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#ifdef USE_CONTRIB
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/xfeatures2d.hpp>
#endif

#define DESCRIPTOR_SIZE_BITS 256
typedef srrg_hbst::BinaryMatchable<cv::KeyPoint, DESCRIPTOR_SIZE_BITS> Matchable;
typedef srrg_hbst::BinaryNode<Matchable> HBSTNode;
typedef srrg_hbst::BinaryTree<HBSTNode> HBSTTree;

// ds nasty global buffers
cv::Ptr<cv::FeatureDetector> keypoint_detector;
cv::Ptr<cv::DescriptorExtractor> descriptor_extractor;
HBSTTree tree;

uint32_t number_of_processed_images = 0;
uint64_t number_of_stored_descriptors = 0;
double image_display_scale = 1;
double maximum_descriptor_distance = 50;
uint32_t number_of_images_interspace = 100;

int32_t main() {
	cv::VideoCapture video("../../../Dataset/KITTI/sequence00//KITTI1_3_3.avi");
	int width = (int)video.get(cv::CAP_PROP_FRAME_WIDTH);
	int height = (int)video.get(cv::CAP_PROP_FRAME_HEIGHT);
	int video_size = (int)video.get(cv::CAP_PROP_FRAME_COUNT) - 2;
	int frame_rate = (int)video.get(cv::CAP_PROP_FPS);

	keypoint_detector = cv::ORB::create(1000);
	descriptor_extractor = cv::ORB::create();

	ofstream fout("result3.txt");
	cout << "timer start ...." << endl;
	auto t_start = std::chrono::high_resolution_clock::now();

	for(int num_count =0; num_count< video_size; num_count++){
		cv::Mat image_read = read_image(video, num_count);
		//std::cout << image_read.type();
		std::cout << num_count << std::endl;
		cv::Mat image;
		
		cv::cvtColor(image_read, image, CV_BGR2GRAY);
		auto hbst_start = std::chrono::high_resolution_clock::now();
		auto hbst_feature_extraction_start = std::chrono::high_resolution_clock::now();
		//std::cout<<image.type();
		//dst.convertTo(src, CV_8UC1);
		// ds detect FAST keypoints
		std::vector<cv::KeyPoint> keypoints;
		keypoint_detector->detect(image, keypoints);

		// ds compute BRIEF descriptors
		cv::Mat descriptors;
		descriptor_extractor->compute(image, keypoints, descriptors);

		// ds obtain linked matchables
		const HBSTTree::MatchableVector matchables(
			HBSTTree::getMatchables(descriptors, keypoints, number_of_processed_images));

		// ds obtain matches against all inserted matchables (i.e. images so far) and integrate them
		// simultaneously
		HBSTTree::MatchVectorMap matches_per_image;
		std::chrono::time_point<std::chrono::system_clock> time_begin(std::chrono::system_clock::now());
		//HBSTTree tree_time;
		//tree_time.matchAndAdd(matchables, matches_per_image, maximum_descriptor_distance);
		//auto hbst_feature_extraction_end = std::chrono::high_resolution_clock::now();
		//fout << double(std::chrono::duration_cast<std::chrono::milliseconds>(hbst_feature_extraction_end - hbst_feature_extraction_start).count()) << "\n";
		//const double processing_duration_seconds = std::chrono::duration<double>(std::chrono::system_clock::now() - time_begin).count();


		tree.matchAndAdd(matchables, matches_per_image, maximum_descriptor_distance);
		const double processing_duration_seconds = std::chrono::duration<double>(std::chrono::system_clock::now() - time_begin).count();
		number_of_stored_descriptors += descriptors.rows;
		++number_of_processed_images;


		// ds info display
		//cv::Mat image_display(image);
		//cv::cvtColor(image_display, image_display, CV_GRAY2RGB);
		
		// ds draw current keypoints in blue
		/*for (const cv::KeyPoint& keypoint : keypoints) {
			cv::circle(image_display, keypoint.pt, 2, cv::Scalar(255, 0, 0), -1);
		}
		*/
		// ds for each match vector (i.e. matching results for each past image) of ALL past images
		if (number_of_processed_images > number_of_images_interspace) {
			for (uint32_t image_number_reference = 0;
				image_number_reference < number_of_processed_images - number_of_images_interspace - 1;
				++image_number_reference) {
				// ds if we have sufficient matches
				const uint32_t number_of_matches = matches_per_image[image_number_reference].size();
				if (number_of_matches > 30) {
					// ds draw matched descriptors
					//fout << "t1=" << number_of_processed_images << '\t' << "coincides with t2=" << image_number_reference << "\n";
					std::cout << "loop closure detected" << std::endl;
					
				}
			}
		}
		

		// ds display image
/*
		const cv::Size current_size(image_display_scale * image_display.cols,
			image_display_scale * image_display.rows);
		cv::resize(image_display, image_display, current_size);
		cv::imshow("current image", image_display);
		cv::waitKey(1);
		*/
		auto hbst_end = std::chrono::high_resolution_clock::now();
		fout << double(std::chrono::duration_cast<std::chrono::milliseconds>(hbst_end - hbst_start).count()) << "\n";
		//ds stats
		std::printf("currentImage|processed images: %6u descriptors: %5d (total: %9lu) processing "
			"time(s): %4.3f\r",
			number_of_processed_images,
			descriptors.rows,
			number_of_stored_descriptors,
			processing_duration_seconds);
		std::fflush(stdout);
		

	}
	fout.close();

	auto t_end = std::chrono::high_resolution_clock::now();
	cout << "dbow time=" << double(std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count()) / 1000.0 << " s" << endl;

	cout << "end" << endl;
	tree.clear(true);
	number_of_processed_images = 0;
	number_of_stored_descriptors = 0;
	while (1);
  return 0;
}

