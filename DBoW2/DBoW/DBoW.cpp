/**
 * File: Demo.cpp
 * Date: November 2011
 * Author: Dorian Galvez-Lopez
 * Description: demo application of DBoW2
 * License: see the LICENSE.txt file
 */

#include <iostream>
#include <vector>

 // DBoW2
#include "DBoW2/DBoW2.h" // defines OrbVocabulary and OrbDatabase

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <chrono>
using namespace DBoW2;
using namespace std;


const int k = 9;
const int L = 5;
const WeightingType weight = TF_IDF;
const ScoringType score = L1_NORM;

#define DBOW_THRESHOLD 0.09  //15


// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

class MyQueryResult
{
public:
	vector<int> id;
	vector<double> similarity;
	MyQueryResult() {};
	~MyQueryResult() {};

	void add(int num, double sim) {
		id.push_back(num);
		similarity.push_back(sim);
	}


private:

};

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
//build and save small vol
void build_vocabulary(vector<vector<cv::Mat > > features);
void changeStructure(const cv::Mat &plain, vector<cv::Mat> &out);
MyQueryResult query_database(OrbDatabase &db, vector<cv::Mat > feature);
cv::Mat read_image(cv::VideoCapture &video, int number);
// ----------------------------------------------------------------------------

int main()
{
	cv::VideoCapture video("../../../Dataset/KITTI/sequence00/KITTI1_grey.avi");
	int width = (int)video.get(cv::CAP_PROP_FRAME_WIDTH);
	int height = (int)video.get(cv::CAP_PROP_FRAME_HEIGHT);
	int video_size = (int)video.get(cv::CAP_PROP_FRAME_COUNT);
	double frame_rate = 10;//(double)video.get(cv::CAP_PROP_FPS);
	
	//1. build feature vector
	vector<vector<cv::Mat > > features;
	features.clear();
	features.reserve(video_size);

	OrbVocabulary voc("small_voc.yml.gz");
	OrbDatabase db(voc, false, 0); //
	ofstream fout("result.txt");
	cv::Ptr<cv::ORB> orb = cv::ORB::create(1000);
	cout << "timer start ...." << endl;
	auto t_start = std::chrono::high_resolution_clock::now();
	cout << "Extracting ORB features..." << endl;
	
	for (int i = 0; i < video_size; i++)
	{
		cout << i << endl;

		cv::Mat image = read_image(video, i);
		auto feature_extraction_start = std::chrono::high_resolution_clock::now();
		cv::Mat mask;
		vector<cv::KeyPoint> keypoints;
		cv::Mat descriptors;

		orb->detectAndCompute(image, mask, keypoints, descriptors);

		features.push_back(vector<cv::Mat >());
		changeStructure(descriptors, features.back());
		
		db.add(features[i]);
		auto feature_extraction_end = std::chrono::high_resolution_clock::now();
		fout << double(std::chrono::duration_cast<std::chrono::milliseconds>(feature_extraction_end - feature_extraction_start).count()) << "\n";
	}
	//fout.close();
	cout << "EXTRACTION DONE" << endl;
	//2 . build database
	
	//build_vocabulary(features);
	/*
	
	OrbVocabulary voc(k, L, weight, score);
	voc.create(features);
	voc.save("small_voc.yml.gz");
	cout << "database built" << endl;
	*/
	//3. loop detection
	


	for (int i = 0; i < video_size; i++) {
		cout << i << endl;
		auto dbow2_start = std::chrono::high_resolution_clock::now();
		MyQueryResult result = query_database(db, features[i]);
		if (result.id.size() > 0)
		{
			for (int j = 0; j < result.id.size(); j++)
			{
				//if (abs(i - result.id[j]) > frame_rate * 10)
				//	fout << "t1=" << (i + 1) / frame_rate << '\t' << "coincides with t2=" << (result.id[j] + 1) / frame_rate << '\t' << "frame1:" << i + 1 << '\t' << "frame2:" << result.id[j] + 1 << " with img1=" << i + 1 << " img2=" << result.id[j] + 1 << "\n";
			}

		}

		db.add(features[i]);
		
		auto dbow2_end = std::chrono::high_resolution_clock::now();
		//fout << double(std::chrono::duration_cast<std::chrono::milliseconds>(dbow2_end - dbow2_start).count()) + average_time << "\n";
	}
	fout.close();
	

	/*
	BowVector v1, v2;
	for (int i = 0; i < video_size; i++)
	{
		voc.transform(features[i], v1);
		for (int j = 0; j < video_size; j++)
		{
			voc.transform(features[j], v2);

			double score = voc.score(v1, v2);
			cout << "Image " << i << " vs Image " << j << ": " << score << endl;
		}
	}
	*/
	// save the vocabulary to disk
	auto t_end = std::chrono::high_resolution_clock::now();
	cout << "dbow time=" << double(std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count()) / 1000.0 << " s" << endl;


	return 0;
}

void changeStructure(const cv::Mat &plain, vector<cv::Mat> &out)
{
	out.resize(plain.rows);

	for (int i = 0; i < plain.rows; ++i)
	{
		out[i] = plain.row(i);
	}
}

//build and save small vol
void build_vocabulary(vector<vector<cv::Mat > > features) {
	OrbVocabulary voc(k, L, weight, score);
	voc.create(features);
	voc.save("small_voc.yml.gz");
}

MyQueryResult query_database(OrbDatabase &db, vector<cv::Mat > feature) {

	QueryResults ret;

	db.query(feature, ret, 4);

	MyQueryResult result;
	for (int i = 0; i < ret.size(); i++)
	{
		if (ret[i].Score > DBOW_THRESHOLD)
		{
			result.id.push_back(ret[i].Id);
			result.similarity.push_back(ret[i].Score);
		}
	}
	return result;
}

cv::Mat read_image(cv::VideoCapture &video, int number) {
	video.set(cv::CAP_PROP_POS_FRAMES, number);
	cv::Mat image;
	video.read(image);
	return image;
}