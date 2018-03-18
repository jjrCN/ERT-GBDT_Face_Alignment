#pragma once
#ifndef _UTILIS_HPP_
#define _UTILIS_HPP_

#include <sys/types.h>
#include <sys/stat.h>
#include <cmath>
#include <sstream>
#include <ctime>
#include <unistd.h>
#include <dirent.h>
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

class sample{
public:
	std::string image_name;
	cv::Mat_<uchar> image;
	cv::Rect GTBox;

	cv::Mat_<float> landmarks_truth;
	cv::Mat_<float> landmarks_truth_normalizaiotn;
	cv::Mat_<float> landmarks_cur;
	cv::Mat_<float> landmarks_cur_normalization;
	cv::Mat_<float> scale_rotate_normalization;
	cv::Mat_<float> transform_normalization;
	cv::Mat_<float> scale_rotate_unnormalization;
	cv::Mat_<float> transform_unnormalization;

	cv::Mat_<float> scale_rotate_from_mean;
	cv::Mat_<float> transform_from_mean;

	int tree_index;
};

class UnLeafNode{
public:
	int landmark_index1;
	int landmark_index2;
	float index1_offset_x;
	float index1_offset_y;
	float index2_offset_x;
	float index2_offset_y;
	float threshold;
};

class TreeModel{
public:
	std::vector<UnLeafNode> splite_model;
	std::vector<cv::Mat_<float>> residual_model;
};

void cut_name(std::string &name, const std::string &name_with_info);

void getfiles(std::vector<std::string> &names, const std::string &path);

bool IsDetected(const cv::Rect &box, const float &x_max, const float &x_min, const float &y_max, const float &y_min);

void Loadimages(std::vector<sample> &data, const std::string &path);

void output(const sample &data, const std::string &path);

void GenerateTraindata(std::vector<sample> &data, const int &initialization);

void GenerateValidationdata(std::vector<sample> &data, const cv::Mat_<float> &global_mean_landmarks);

void compute_similarity_transform(const cv::Mat_<float> &target, const cv::Mat_<float> &origin, cv::Mat_<float> &scale_rotate, cv::Mat_<float> &transform);

void normalization(cv::Mat_<float> &target, const cv::Mat_<float> &origin, const cv::Mat_<float> &scale_rotate, const cv::Mat_<float> &transform);

void check_edge(sample &data);

float compute_Error(const std::vector<sample> &data);

#endif