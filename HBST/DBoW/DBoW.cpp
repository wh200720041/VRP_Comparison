/**
 * Date:  2016
 * Author: Rafael Muñoz Salinas
 * Description: demo application of DBoW3
 * License: see the LICENSE.txt file
 */

//#define USE_CONTRIB

#include "Util.h"
#include "test.h"
#include "DescManip.h"
#include <opencv2/imgproc/imgproc.hpp>
/*
我的一些理解：

用sliding window 去去除一下unnecessary feature points
然后再把这些信息加入数据库中
方法1：增强robostness， 可以跟dbow对比

*/
const int sliding_window_size = 1;//total size 

//#define USE_MY_DBOW3


int main(int argc, char **argv)
{

	cv::VideoCapture video("C:\\Users\\IceHan\\Desktop\\LoopClosureCpp\\MyDBoW3\\DBoW\\video\\KITTI1.avi");
	int width = (int)video.get(cv::CAP_PROP_FRAME_WIDTH);
	int height = (int)video.get(cv::CAP_PROP_FRAME_HEIGHT);
	int video_size = (int)video.get(cv::CAP_PROP_FRAME_COUNT) - 2;
	int frame_rate = (int)video.get(cv::CAP_PROP_FPS);


	//image_check_test(video);
	//bow_test(video);
	//img_test(video);
	//similarity_test(video);

	//build_vocabulary(video);
	vector<cv::Mat> descriptor_arr;
	vector<vector<KeyPoint>> keypoint_arr;
	std::vector<KeyPoint> keypoints_temp;
	cv::Mat descriptors_temp;
	cv::Ptr<cv::Feature2D> fdetector = cv::ORB::create(FEATURE_NUMBER);//cv::BRISK::create();cv::AKAZE::create();cv::xfeatures2d::SURF::create(400, 4, 2, EXTENDED_SURF);
	
	Vocabulary voc("lib\\ORBvoc.txt");
	//Vocabulary voc("voc.yml.gz");

	Database db(voc, false, 0);


	ofstream fout("result.txt");
	cout << "timer start ...." << endl;
	auto t_start = std::chrono::high_resolution_clock::now();
	
	cv::Mat image = read_image(video, 1);

	cv::Mat image2;
	cv::cvtColor(image, image2, CV_BGR2GRAY);
	keypoints_temp.clear();
	descriptors_temp.release();

	/*
	float data[96*3] = {
		2.5, 2.4, 0.5, 0.7,2.5, 2.4, 0.5, 0.7, 2.5, 2.4, 0.5, 0.7, 2.5, 2.4, 0.5, 0.7, 2.5, 2.4, 0.5, 0.7, 2.5, 2.4, 0.5, 0.7, 2.5, 2.4, 0.5, 0.7, 2.5, 2.4, 0.5, 0.7, 2.5, 2.4, 0.5, 0.7, 2.5, 2.4, 0.5, 0.7, 2.5, 2.4, 0.5, 0.7, 2.5, 2.4, 0.5, 0.7, 2.5, 2.4, 0.5, 0.7, 2.5, 2.4, 0.5, 0.7, 2.5, 2.4, 0.5, 0.7, 2.5, 2.4, 0.5, 0.7, 2.5, 2.4, 0.5, 0.7, 2.5, 2.4, 0.5, 0.7, 2.5, 2.4, 0.5, 0.7, 2.5, 2.4, 0.5, 0.7, 2.5, 2.4, 0.5, 0.7, 2.5, 2.4, 0.5, 0.7, 2.5, 2.4, 0.5, 0.7, 2.5, 2.4, 0.5, 0.7,
		2.5, 2.4, 0.5, 0.7, 2.5, 2.4, 0.5, 0.7, 2.5, 2.4, 0.5, 0.7, 2.5, 2.4, 0.5, 0.7, 2.5, 2.4, 0.5, 0.7, 2.5, 2.4, 0.5, 0.7, 2.5, 2.4, 0.5, 0.7, 2.5, 2.4, 0.5, 0.7, 2.5, 2.4, 0.5, 0.7, 2.5, 2.4, 0.5, 0.7, 2.5, 2.4, 0.5, 0.7, 2.5, 2.4, 0.5, 0.7, 2.5, 2.4, 0.5, 0.7, 2.5, 2.4, 0.5, 0.7, 2.5, 2.4, 0.5, 0.7, 2.5, 2.4, 0.5, 0.7, 2.5, 2.4, 0.5, 0.7, 2.5, 2.4, 0.5, 0.7, 2.5, 2.4, 0.5, 0.7, 2.5, 2.4, 0.5, 0.7, 2.5, 2.4, 0.5, 0.7, 2.5, 2.4, 0.5, 0.7, 2.5, 2.4, 0.5, 0.7, 2.5, 2.4, 0.5, 0.7,
		2.5, 2.4, 0.5, 0.7, 2.5, 2.4, 0.5, 0.7, 2.5, 2.4, 0.5, 0.7, 2.5, 2.4, 0.5, 0.7, 2.5, 2.4, 0.5, 0.7, 2.5, 2.4, 0.5, 0.7, 2.5, 2.4, 0.5, 0.7, 2.5, 2.4, 0.5, 0.7, 2.5, 2.4, 0.5, 0.7, 2.5, 2.4, 0.5, 0.7, 2.5, 2.4, 0.5, 0.7, 2.5, 2.4, 0.5, 0.7, 2.5, 2.4, 0.5, 0.7, 2.5, 2.4, 0.5, 0.7, 2.5, 2.4, 0.5, 0.7, 2.5, 2.4, 0.5, 0.7, 2.5, 2.4, 0.5, 0.7, 2.5, 2.4, 0.5, 0.7, 2.5, 2.4, 0.5, 0.7, 2.5, 2.4, 0.5, 0.7, 2.5, 2.4, 0.5, 0.7, 2.5, 2.4, 0.5, 0.7, 2.5, 2.4, 0.5, 0.7, 2.5, 2.4, 0.5, 0.7,
	};
	cv::Mat your_matrix = cv::Mat(1, 96 * 3, CV_32F, data);
			for (int k=0;k<2000;k++){
			int numOfComponents = 10;
			PCA pca(your_matrix, cv::Mat(), CV_PCA_DATA_AS_ROW, numOfComponents);
		}
	*/

	for (int i = 0; i < video_size; i++) {
		cv::Mat image = read_image(video, i);

		/*
		fdetector->detectAndCompute(image2, cv::Mat(), keypoints_temp, descriptors_temp);
		
		cv::Mat image2;
		cv::cvtColor(image, image2, CV_BGR2GRAY);
		Mat padded;
		int m = getOptimalDFTSize(image2.rows);
		int n = getOptimalDFTSize(image2.cols); // on the border add zero values
		copyMakeBorder(image2, padded, 0, m - image2.rows, 0, n - image2.cols, BORDER_CONSTANT, Scalar::all(0));
		Mat planes[] = { Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F) };
		Mat complexI;
		merge(planes, 2, complexI);         // Add to the expanded another plane with zeros
		dft(complexI, complexI);            // this way the result may fit in the source matrix
		// compute the magnitude and switch to logarithmic scale
		// => log(1 + sqrt(Re(DFT(I))^2 + Im(DFT(I))^2))
		split(complexI, planes);                   // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
		magnitude(planes[0], planes[1], planes[0]);// planes[0] = magnitude
		Mat magI = planes[0];
		magI += Scalar::all(1);                    // switch to logarithmic scale
		log(magI, magI);
		*/

		
#ifdef USE_MY_DBOW3
		//extractor descriptors
		keypoints_temp.clear();
		descriptors_temp.release();
		fdetector->detectAndCompute(image, cv::Mat(), keypoints_temp, descriptors_temp);


		if (i >= sliding_window_size) {
			descriptor_arr.pop_back();
			keypoint_arr.pop_back();
		}
		descriptor_arr.insert(descriptor_arr.begin(), descriptors_temp);
		keypoint_arr.insert(keypoint_arr.begin(), keypoints_temp);
		//compute descriptor

		//cv::Mat final_descriptor = extractDiscriptorWithMinSize(descriptor_arr, keypoint_arr,100);
		cv::Mat final_descriptor = extractDiscriptorWithFixedSize(descriptor_arr, keypoint_arr,2000);
		//cv::Mat final_descriptor = extractDiscriptorWithExtra(descriptor_arr, keypoint_arr);
		//cv::Mat final_descriptor = extractDiscriptor(descriptor_arr, keypoint_arr);
#else
		keypoints_temp.clear();
		cv::Mat final_descriptor;
		fdetector->detectAndCompute(image, cv::Mat(), keypoints_temp, final_descriptor);
#endif // USE_MY_DBOW3
		
		

		QueryResults ret;
		db.query(final_descriptor, ret, -1, (std::max)(i - frame_rate * 10,0));//这里的数字指的是留下来的max similarity的个数
		
		

		if (ret.size() > 0)
		{
			for (int j = 0; j < ret.size(); j++)
			{
				if (ret[j].Score > DBOW_THRESHOLD) {
					fout << "t1=" << i << '\t' << "coincides with t2=" << ret[j].Id  << "\n";
					cout << "loop detected" << endl;
					break;
				}					
			}
		}
		
		db.add(final_descriptor);
		
		cout << i << endl;

	}
	
	fout.close();

	auto t_end = std::chrono::high_resolution_clock::now();
	cout << "dbow time=" << double(std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count()) / 1000.0 << " s" << endl;

	cout << "end" << endl;
	
	return 0;
}


