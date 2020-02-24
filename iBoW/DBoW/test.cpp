
#include "Util.h"
#include "DescriptorFilter.h"

void similarity_test(cv::VideoCapture &video) {
	cv::Mat image1 = read_image(video, 464);
	cv::Mat image2 = read_image(video, 1230);

	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
	vector<DMatch> matches;
	cv::Mat descriptor1;
	cv::Mat descriptor2;
	vector<KeyPoint> keypoint1;
	vector<KeyPoint> keypoint2;
	cv::Ptr<cv::Feature2D> fdetector = cv::ORB::create();//cv::BRISK::create();cv::AKAZE::create();cv::xfeatures2d::SURF::create(400, 4, 2, EXTENDED_SURF);

	fdetector->detectAndCompute(image1, cv::Mat(), keypoint1, descriptor1);
	fdetector->detectAndCompute(image2, cv::Mat(), keypoint2, descriptor2);
	matcher->match(descriptor1, descriptor2, matches);
	Vocabulary voc("lib\\ORBvoc.txt");
	//Vocabulary voc("voc.yml.gz");
	Database db(voc, false, 0);

	QueryResults ret;
	db.add(descriptor1);
	db.query(descriptor2, ret, 4);
	cout << "similarity is :" << ret[0].Score << endl;
	return;
}
void img_test(cv::VideoCapture &video) {
	cv::Mat image1 = read_image(video, 464);
	cv::Mat image2 = read_image(video, 1230);

	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
	vector<DMatch> matches;
	cv::Mat descriptor1;
	cv::Mat descriptor2;
	vector<KeyPoint> keypoint1;
	vector<KeyPoint> keypoint2;
	cv::Ptr<cv::Feature2D> fdetector = cv::ORB::create();//cv::BRISK::create();cv::AKAZE::create();cv::xfeatures2d::SURF::create(400, 4, 2, EXTENDED_SURF);

	fdetector->detectAndCompute(image1, cv::Mat(), keypoint1, descriptor1);
	fdetector->detectAndCompute(image2, cv::Mat(), keypoint2, descriptor2);
	matcher->match(descriptor1, descriptor2, matches);




	auto t_start = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < 10000; i++) {
		vector<KeyPoint> keypoint1_temp;
		fdetector->detect(image1, keypoint1_temp);
		//fdetector->compute(image1, keypoint1, descriptor_temp);
	}

	auto t_end = std::chrono::high_resolution_clock::now();
	cout << "dbow time=" << double(std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count()) / 1000.0 << " s" << endl;




	Mat img_match;
	Mat img_goodmatch;
	drawMatches(image1, keypoint1, image2, keypoint2, matches, img_match);
	double min_dist = 10000;
	for (int j = 0; j < descriptor1.rows; j++)
	{
		double dist = matches[j].distance;
		if (dist < min_dist) min_dist = dist;
	}
	std::vector< DMatch > good_matches;



	//descriptor2.push_back();

	vector<KeyPoint> keypoint_final1, keypoint_final2;
	cv::Mat descriptor_final1, dst;
	cv::Mat descriptor_final2;

	for (int j = 0; j < descriptor1.rows; j++)
	{
		if (matches[j].distance <= max(2 * min_dist, 50.0))
		{
			good_matches.push_back(matches[j]);
			keypoint_final1.push_back(keypoint1[matches[j].queryIdx]);
			descriptor_final2.push_back(descriptor1.row(matches[j].queryIdx));
		}
	}

	std::cout << "good match point " << ":" << good_matches.size() << endl;

	fdetector->compute(image1, keypoint_final1, descriptor_final1);


	cv::bitwise_xor(descriptor_final1, descriptor_final2, dst);
	if (cv::countNonZero(dst) > 0)
		cout << "not equal" << endl;

	drawMatches(image1, keypoint1, image2, keypoint2, good_matches, img_goodmatch);
	//cout << "good matches:" << good_matches.size() << endl;
	imshow("所有匹配点对", img_goodmatch);
	//imshow("优化后匹配点对", img_goodmatch);
	waitKey(30);
	//
	return;
}

void bow_test(cv::VideoCapture &video) {
	cv::Mat image1 = read_image(video, 1313);//1313
	cv::Mat image2 = read_image(video, 1315);//1315

	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
	vector<DMatch> matches;
	cv::Mat descriptor1;
	cv::Mat descriptor2;
	cv::Mat descriptor3;
	cv::Mat descriptor4;
	cv::Mat image;
	vector<KeyPoint> keypoint1;
	vector<KeyPoint> keypoint2;
	cv::Ptr<cv::Feature2D> fdetector = cv::ORB::create();//cv::BRISK::create();cv::AKAZE::create();cv::xfeatures2d::SURF::create(400, 4, 2, EXTENDED_SURF);

	fdetector->detectAndCompute(image1, cv::Mat(), keypoint1, descriptor1);
	fdetector->detectAndCompute(image2, cv::Mat(), keypoint2, descriptor2);
	matcher->match(descriptor1, descriptor2, matches);
	drawMatches(image1, keypoint1, image2, keypoint2, matches, image);
	//cout << "good matches:" << good_matches.size() << endl;
	//imshow("所有匹配点对", image);
	//waitKey(30);
	Vocabulary voc("lib\\ORBvoc.txt");
	//Vocabulary voc("voc.yml.gz");

	Database db(voc, false, 0);

	descriptor3.push_back(descriptor1.row(1));

	descriptor3.push_back(descriptor1.row(2));
	descriptor3.push_back(descriptor1.row(3));
	descriptor4.push_back(descriptor1.row(1));
	db.add(descriptor3);
	db.add(descriptor4);

	QueryResults ret1;
	db.query(descriptor4, ret1, -1);//这里的数字指的是留下来的max similarity的个数
	cout << endl;

	return;
}

void image_check_test(cv::VideoCapture &video) {
	cv::Mat image1 = read_image(video, 3055);//1313
	cv::Mat image2 = read_image(video, 3054);//1315
	cv::Mat image3 = read_image(video, 3053);//1313
	cv::Mat image4 = read_image(video, 256);//1313
	cv::Mat image5 = read_image(video, 255);//1313
	cv::Mat image6 = read_image(video, 254);//1313

	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
	vector<DMatch> matches;
	cv::Mat descriptor1, descriptor2, descriptor3, descriptor4, descriptor5, descriptor6;
	cv::Mat image;
	vector<KeyPoint> keypoint1, keypoint2, keypoint3, keypoint4, keypoint5, keypoint6;
	

	get_keypoint_and_descriptor(image1, keypoint1, descriptor1);
	get_keypoint_and_descriptor(image2, keypoint2, descriptor2);
	get_keypoint_and_descriptor(image3, keypoint3, descriptor3);
	get_keypoint_and_descriptor(image4, keypoint4, descriptor4);
	get_keypoint_and_descriptor(image5, keypoint5, descriptor5);
	get_keypoint_and_descriptor(image6, keypoint6, descriptor6);

	matcher->match(descriptor1, descriptor4, matches);
	//drawMatches(image1, keypoint1, image2, keypoint2, matches, image);
	//cout << "good matches:" << good_matches.size() << endl;
	//imshow("所有匹配点对", image);
	//waitKey(30);

	double min_dist = 10000;
	for (int j = 0; j < descriptor1.rows; j++)
	{
		double dist = matches[j].distance;
		if (dist < min_dist) min_dist = dist;
	}
	std::vector< DMatch > good_matches;

	vector<KeyPoint> keypoint_final1, keypoint_final2;
	cv::Mat descriptor_final1;
	cv::Mat descriptor_final2;

	for (int j = 0; j < descriptor1.rows; j++)
	{
		if (matches[j].distance <= max(2 * min_dist, 50.0))
		{
			good_matches.push_back(matches[j]);
			//keypoint_final1.push_back(keypoint1[matches[j].queryIdx]);
			descriptor_final2.push_back(descriptor1.row(matches[j].queryIdx));
			descriptor_final1.push_back(descriptor4.row(matches[j].trainIdx));
		}
	}

	cout << "good matches:" << good_matches.size() << endl;

	imshow("所有匹配点1对", image1);
	
	imshow("所有匹配点2对", image4);
	waitKey(0);
	display_matched_image(image1, keypoint1, image4, keypoint2, good_matches);

	Vocabulary voc("lib\\ORBvoc.txt");
	Database db(voc, false, 0);

	vector<cv::Mat> descriptor_arr;
	vector<vector<KeyPoint>> keypoint_arr;
	descriptor_arr.insert(descriptor_arr.begin(), descriptor3);
	descriptor_arr.insert(descriptor_arr.begin(), descriptor2);
	descriptor_arr.insert(descriptor_arr.begin(), descriptor1);
	keypoint_arr.insert(keypoint_arr.begin(), keypoint3);
	keypoint_arr.insert(keypoint_arr.begin(), keypoint2);
	keypoint_arr.insert(keypoint_arr.begin(), keypoint1);
	cv::Mat descriptor_1 = extractDiscriptorWithExtra(descriptor_arr, keypoint_arr);
	db.add(descriptor_1);

	descriptor_arr.clear();
	keypoint_arr.clear();
	descriptor_arr.insert(descriptor_arr.begin(), descriptor6);
	descriptor_arr.insert(descriptor_arr.begin(), descriptor5);
	descriptor_arr.insert(descriptor_arr.begin(), descriptor4);
	keypoint_arr.insert(keypoint_arr.begin(), keypoint6);
	keypoint_arr.insert(keypoint_arr.begin(), keypoint5);
	keypoint_arr.insert(keypoint_arr.begin(), keypoint4);
	cv::Mat descriptor_2 = extractDiscriptorWithExtra(descriptor_arr, keypoint_arr);
	db.add(descriptor_2);

	QueryResults ret;
	db.query(descriptor_2, ret, -1);//这里的数字指的是留下来的max similarity的个数
	cout << endl;

	cout << endl;

	return;

}