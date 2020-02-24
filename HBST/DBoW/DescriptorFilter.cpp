#include "DescriptorFilter.h"


/*
	filter out the unnecessary descriptor
	the [0] is the current frame
*/
cv::Mat extractDiscriptor(vector<Mat> &descriptor, vector<vector<KeyPoint>> &keypoint) {

	if (descriptor.size() == 0)
		throw("descriptor size is 0");

	int window_size = descriptor.size();
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
	bool success_descriptor[feature_size];
	for (int i = 0; i < descriptor[0].rows; i++) {
		success_descriptor[i] = true;
	}


	for (int i = 1; i < window_size; i++) {

		//find the matches 
		vector<DMatch> matches;
		matcher->match(descriptor[0], descriptor[i], matches);
		//find min distance
		double min_dist = 10000;
		//找出所有匹配之间的最小距离和最大距离, 即是最相似的和最不相似的两组点之间的距离

		for (int j = 0; j < descriptor[0].rows; j++)
		{
			double dist = matches[j].distance;
			if (dist < min_dist) min_dist = dist;
			//if (dist > max_dist) max_dist = dist;
		}
		//std::vector< DMatch > good_matches;

		for (int j = 0; j < descriptor[i].rows; j++)
		{
			if (matches[j].distance > max(2 * min_dist, 30.0))
			{
				//unqualified, remove
				success_descriptor[j] = false;
				//good_matches.push_back(matches[i]);
			}
		}
		//std::cout << "match point " << i << ":" << good_matches.size() << endl;
	}

	//std::cout << "match point " << ":" << success_descriptor.size() << endl;

	cv::Mat output_descriptor;
	//descriptor[0].copyTo(output_descriptor);

	for (int i = 0; i < descriptor[0].rows; i++) {
		if (success_descriptor[i] == true)
			output_descriptor.push_back((descriptor[0]).row(i));
	}

	cout << "filtered points: " << output_descriptor.rows << endl;
	return output_descriptor;
}


cv::Mat extractDiscriptorWithFixedSize(vector<Mat> &descriptor, vector<vector<KeyPoint>> &keypoint, int size) {
	if (descriptor.size() == 0)
		throw("descriptor size is 0");
	else if (descriptor.size() == 1) {
		cout << "size wrong ! filtered points: " << descriptor[0].rows << endl;
		return descriptor[0];
	}
	int window_size = descriptor.size();
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
	
	vector<double> descriptor_distance;
	for (int i = 0; i < descriptor[0].rows; i++) {
		descriptor_distance.push_back(0.0);
	}

	for (int i = 1; i < window_size; i++) {
		//find the matches 
		vector<DMatch> matches;
		matcher->match(descriptor[0], descriptor[i], matches);
		//find min distance
		double min_dist = 10000;
		//找出所有匹配之间的最小距离和最大距离, 即是最相似的和最不相似的两组点之间的距离

		for (int j = 0; j < descriptor[0].rows; j++)
		{
			descriptor_distance[j] += matches[j].distance;
		}
		//std::vector< DMatch > good_matches;
	}


	//std::cout << "match point " << ":" << success_descriptor.size() << endl;

	cv::Mat output_descriptor;
	vector<double> descriptor_distance_sorted = descriptor_distance;

	//descriptor[0].copyTo(output_descriptor);
	sort(descriptor_distance_sorted.begin(), descriptor_distance_sorted.end());


	for (int i = 0; i < descriptor[0].rows; i++) {
		if(descriptor_distance[i]< descriptor_distance_sorted[size])
			output_descriptor.push_back((descriptor[0]).row(i));
	}

	cout << "filtered points: " << output_descriptor.rows << endl;
	return output_descriptor;


}

cv::Mat extractDiscriptorWithMinSize(vector<Mat> &descriptor, vector<vector<KeyPoint>> &keypoint, int min_size) {

	if (descriptor.size() == 0)
		throw("descriptor size is 0");

	int window_size = descriptor.size();
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
	bool success_descriptor[feature_size];
	for (int i = 0; i < descriptor[0].rows; i++) {
		success_descriptor[i] = true;
	}

	vector<double> descriptor_distance;
	for (int i = 0; i < descriptor[0].rows; i++) {
		descriptor_distance.push_back(0.0);
	}

	for (int i = 1; i < window_size; i++) {
		//find the matches 
		vector<DMatch> matches;
		matcher->match(descriptor[0], descriptor[i], matches);
		//find min distance
		double min_dist = 10000;
		//找出所有匹配之间的最小距离和最大距离, 即是最相似的和最不相似的两组点之间的距离

		for (int j = 0; j < descriptor[0].rows; j++)
		{
			descriptor_distance[j] += matches[j].distance;
		}
		//std::vector< DMatch > good_matches;
	}

	//std::vector< DMatch > good_matches;

	double min_dist = 1000;
	for (int i = 0; i < descriptor[0].rows; i++)
	{
		double dist = descriptor_distance[i];
		if (dist < min_dist) min_dist = dist;
		//if (dist > max_dist) max_dist = dist;
	}
	//std::vector< DMatch > good_matches;

	for (int i = 0; i < descriptor[0].rows; i++)
	{
		if (descriptor_distance[i] > max(2 * min_dist, 30.0 ))
		{
			//unqualified, remove
			success_descriptor[i] = false;
		}
	}

	cv::Mat output_descriptor;
	//descriptor[0].copyTo(output_descriptor);

	for (int i = 0; i < descriptor[0].rows; i++) {
		if (success_descriptor[i] == true)
			output_descriptor.push_back((descriptor[0]).row(i));
	}

	if (output_descriptor.rows < min_size) {
		int start = output_descriptor.rows;
		for (int i = start; i < min_size; i++) {
			
			double min_distance = 1000;
			int min_id = 1;
			for (int j = 0; j < descriptor[0].rows; j++) {
				if (success_descriptor[i] == false && descriptor_distance[j]< min_distance) {
					min_distance = descriptor_distance[j];
					min_id = j;
				}
				
			}
			success_descriptor[min_id] = true;
			//output_descriptor.push_back((descriptor[0]).row(min_id));
		}
	}


	cout << "filtered points: " << output_descriptor.rows << endl;
	return output_descriptor;
}


cv::Mat extractDiscriptorWithExtra(vector<Mat> &descriptor, vector<vector<KeyPoint>> &keypoint) {

	if (descriptor.size() == 0)
		throw("descriptor size is 0");

	int window_size = descriptor.size();
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
	bool success_descriptor[feature_size];
	cv::Mat output_descriptor;
	descriptor[0].copyTo(output_descriptor);
	for (int i = 0; i < descriptor[0].rows; i++) {
		success_descriptor[i] = true;
	}

	for (int i = 1; i < window_size; i++) {

		//find the matches 
		vector<DMatch> matches;
		matcher->match(descriptor[0], descriptor[i], matches);
		//find min distance
		double min_dist = 10000;
		//找出所有匹配之间的最小距离和最大距离, 即是最相似的和最不相似的两组点之间的距离

		for (int j = 0; j < descriptor[0].rows; j++)
		{
			double dist = matches[j].distance;
			if (dist < min_dist) min_dist = dist;
			//if (dist > max_dist) max_dist = dist;
		}
		//std::vector< DMatch > good_matches;

		for (int j = 0; j < descriptor[i].rows; j++)
		{
			if (matches[j].distance > max(2 * min_dist, 30.0))
			{
				//unqualified, remove
				success_descriptor[j] = false;
				//good_matches.push_back(matches[i]);
			}
		}
	}

	//std::vector< DMatch > good_matches;
 
	//descriptor[0].copyTo(output_descriptor);

	for (int i = 0; i < descriptor[0].rows; i++) {
		if (success_descriptor[i] == true)
			output_descriptor.push_back((descriptor[0]).row(i));
	}

	cout << "filtered points: " << output_descriptor.rows << endl;
	return output_descriptor;
}
 
   