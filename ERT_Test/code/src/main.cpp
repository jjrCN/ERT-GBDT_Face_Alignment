#ifdef WIN32
#define NOMINMAX
#endif
#include <clipp.h>

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
	bool help = false;
	std::string input_filename;
	std::string face_detector_path = "./facedetection/haarcascade_frontalface_alt2.xml";
	std::string model_path = "./result/model/ERT.bin";
    auto cli = (
        clipp::option("-h", "--help").set(help, true),
		(clipp::option("-f", "--face-detector") & clipp::value("face detector xml path").set(face_detector_path)),
		(clipp::option("-m", "--model") & clipp::value("model path").set(model_path)),
        (clipp::option("-i", "--input") & clipp::value("input image filename").set(input_filename))
    );

	auto parse_result = clipp::parse(argc, argv, cli);
    if (!parse_result || help) {
        std::cout << clipp::make_man_page(cli, argv[0]);
        return -1;
    }

	// Model model;
	ERT ert;
	ert.load_binary(model_path);

	cv::CascadeClassifier haar_cascade;
	haar_cascade.load(face_detector_path);
	std::cout << "load face detector completed." << std::endl;

	auto convert_cv_rect = [](const cv::Rect& rect) {
		return Eigen::Vector4f(
			(float)(rect.x),
			(float)(rect.x + rect.width),
			(float)(rect.y),
			(float)(rect.y + rect.height));
	};

	auto draw_landmark_rect = [](cv::Mat& colorImage, const cv::Rect& rect, const Eigen::MatrixX2f& landmark) {
		for(int i = 0; i < (int)landmark.rows(); ++i)
		{
			int x = (int)landmark(i, 0);
			int y = (int)landmark(i, 1);
			cv::circle(colorImage, cv::Point(x, y), 1, cv::Scalar(255, 255, 255), -1);
		}
		cv::rectangle(colorImage, rect, cv::Scalar(255, 255, 255), 1, 1, 0);
	};

	Eigen::MatrixX2f landmark(ert.get_landmark_number(), 2);
	landmark.setZero();

	if (!input_filename.empty()) {
		auto colorImage = cv::imread(input_filename);
		cv::Mat_<uchar> image;
		cv::cvtColor(colorImage, image, cv::COLOR_BGR2GRAY);

  		std::vector<cv::Rect> faces_temp;
		haar_cascade.detectMultiScale(image, faces_temp, 1.1, 2, 0, cv::Size(30, 30));

		if (!faces_temp.empty()) {
			ert.find_landmark(image, convert_cv_rect(faces_temp[0]), landmark);
			draw_landmark_rect(colorImage, faces_temp[0], landmark);
		}

		cv::imshow("face", colorImage);
	   	cv::waitKey(0);

		exit(0);
	}
	else {
		std::cout << "An input file is not specified. Try to open video input..." << std::endl;
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
			cv::Mat colorImage;
			cv::Mat_<uchar> image;
			cap >> colorImage;
			cv::cvtColor(colorImage, image, cv::COLOR_BGR2GRAY);

			bool face_found = false;
			auto bbMin = landmark.colwise().minCoeff();
			auto bbMax = landmark.colwise().maxCoeff();
			auto bbSize = (bbMax - bbMin).prod();
			if (bbSize > 100.0f && bbSize <= (float)(image.rows * image.cols))
				face_found = true;
			if (face_found) {
				ert.find_landmark(image, landmark);
				draw_landmark_rect(colorImage, cv::Rect(0, 0, 1, 1), landmark);
			}
			else {
				std::vector<cv::Rect> faces_temp;
				haar_cascade.detectMultiScale(image, faces_temp, 1.1, 2, 0, cv::Size(30, 30));
				if (!faces_temp.empty()) {
					ert.find_landmark(image, convert_cv_rect(faces_temp[0]), landmark);
					draw_landmark_rect(colorImage, faces_temp[0], landmark);
				}
			}

			cv::imshow("face", colorImage);
			cv::waitKey(1);
		}
	}
}
