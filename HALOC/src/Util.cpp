#include "Util.h"




void build_vocabulary(vector<string> path_to_images) {
	// branching factor and depth levels

	cv::Ptr<cv::Feature2D> fdetector = cv::ORB::create();//cv::BRISK::create();cv::AKAZE::create();cv::xfeatures2d::SURF::create(400, 4, 2, EXTENDED_SURF);
	vector<cv::Mat> features;
	for (size_t i = 0; i < path_to_images.size(); ++i)
	{
		vector<cv::KeyPoint> keypoints;
		cv::Mat descriptors;
		std::cout << "reading image: " << path_to_images[i] << endl;
		cv::Mat image = cv::imread(path_to_images[i], 0);
		if (image.empty())throw std::runtime_error("Could not open image" + path_to_images[i]);
		fdetector->detectAndCompute(image, cv::Mat(), keypoints, descriptors);
		features.push_back(descriptors);
		cout << "done detecting features" << endl;
	}


	DBoW3::Vocabulary voc(k, L, weight, score);

	std::cout << "Creating a small " << k << "^" << L << " vocabulary..." << endl;
	voc.create(features);
	//save vocal
	voc.save("small_voc.yml.gz");
}

//build and save small vol
void build_vocabulary(cv::VideoCapture &video) {
	int video_size = (int)video.get(cv::CAP_PROP_FRAME_COUNT);
	cv::Ptr<cv::Feature2D> fdetector = cv::ORB::create();//cv::BRISK::create();cv::AKAZE::create();cv::xfeatures2d::SURF::create(400, 4, 2, EXTENDED_SURF);
	vector<cv::Mat> features;
	for (int i = 0; i < video_size; i++) {
		cv::Mat image = read_image(video, i);
		vector<cv::KeyPoint> keypoints;
		cv::Mat descriptors;
		fdetector->detectAndCompute(image, cv::Mat(), keypoints, descriptors);
		features.push_back(descriptors);
		std::cout << i << endl;
	}
	DBoW3::Vocabulary voc(k, L, weight, score);
	voc.create(features);
	//save vocal
	voc.save("voc.yml.gz");
}

cv::Mat feature_extraction(cv::Mat image) {
	cv::Ptr<cv::Feature2D> fdetector = cv::ORB::create();//cv::BRISK::create();cv::AKAZE::create();cv::xfeatures2d::SURF::create(400, 4, 2, EXTENDED_SURF);
	vector<cv::KeyPoint> keypoints;
	cv::Mat descriptors;
	fdetector->detectAndCompute(image, cv::Mat(), keypoints, descriptors);
	return descriptors;
}

cv::Mat read_image(cv::VideoCapture &video, int number) {
	video.set(cv::CAP_PROP_POS_FRAMES, number);
	cv::Mat image;
	video.read(image);
	return image;
}

void get_keypoint_and_descriptor(cv::Mat image, vector<KeyPoint>& keypoint, cv::Mat& descriptor) {
	cv::Ptr<cv::Feature2D> fdetector = cv::ORB::create(FEATURE_NUMBER);//cv::BRISK::create();cv::AKAZE::create();cv::xfeatures2d::SURF::create(400, 4, 2, EXTENDED_SURF);
	fdetector->detectAndCompute(image, cv::Mat(), keypoint, descriptor);
	return ;
}

void display_matched_image(cv::Mat image1, vector<KeyPoint> keypoint1, cv::Mat image2, vector<KeyPoint> keypoint2, std::vector< DMatch > matches) {
	cv::Mat image;
	drawMatches(image1, keypoint1, image2, keypoint2, matches, image);
	imshow("所有匹配点对2", image);
	waitKey(0);
}
