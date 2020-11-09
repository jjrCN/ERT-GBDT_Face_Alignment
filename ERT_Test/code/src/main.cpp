#include <cmath>
#include <sstream>
#include <iostream>
#include <fstream>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <Eigen/Dense>

#include <string>
#include <set>
#include <exception>

#include <ERT.hpp>

using namespace ert;

int main(int argc, char* argv[])
{
	// Model model;
	ERT ert;
	ert.load_binary("./result/model/ERT.bin");

	std::string haar_feature = "./facedetection/haarcascade_frontalface_alt2.xml";
	cv::CascadeClassifier haar_cascade;
	haar_cascade.load(haar_feature);
	std::cout << "load face detector completed." << std::endl;

	cv::VideoCapture cap(0);
  	if(!cap.isOpened())
 	{
    	std::cout << "Video open failed. please check your video equitment." << std::endl;
    	exit(0);
  	}
  	std::cout << "open video." << std::endl;
  	int m = 1;
  	while(m)
  	{
		cv::Mat3b colorImage;
		cv::Mat_<uchar> image;
  		cap >> colorImage;
		cv::cvtColor(colorImage, image, cv::COLOR_BGR2GRAY);

  		std::vector<cv::Rect> faces_temp;
		haar_cascade.detectMultiScale(image, faces_temp, 1.1, 2, 0, cv::Size(30, 30));
		if (!faces_temp.empty()) {
			Eigen::Vector4f bbox;
			bbox(0) = (float)(faces_temp[0].x);
			bbox(1) = (float)(faces_temp[0].x + faces_temp[0].width);
			bbox(2) = (float)(faces_temp[0].y);
			bbox(3) = (float)(faces_temp[0].y + faces_temp[0].height);

			Eigen::MatrixX2f landmark;
			ert.find_landmark(image, bbox, landmark);

			for(int i = 0; i < (int)landmark.rows(); ++i)
			{
				int x = (int)landmark(i, 0);
				int y = (int)landmark(i, 1);
				cv::circle(colorImage, cv::Point(x, y), 1, cv::Scalar(255, 255, 255), -1);
			}
			cv::rectangle(colorImage, faces_temp[0], cv::Scalar(255, 255, 255), 1, 1, 0);
		}

		cv::imshow("face", colorImage);
	   	cv::waitKey(1);
  	}
}
