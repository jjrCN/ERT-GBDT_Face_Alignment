#pragma once
#ifndef _UTILIS_HPP_
#define _UTILIS_HPP_

#include <sys/types.h>
#include <sys/stat.h>
#include <cmath>
#include <sstream>
#include <ctime>
// #include <unistd.h>
// #include <dirent.h>
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <Eigen/Dense>

class sample{
public:
	std::string image_name;
	cv::Mat_<uchar> image;
	cv::Rect GTBox;

	Eigen::MatrixX2f landmarks_truth;
	Eigen::MatrixX2f landmarks_truth_normalizaiotn;
	Eigen::MatrixX2f landmarks_cur;
	Eigen::MatrixX2f landmarks_cur_normalization;
	Eigen::Matrix2f scale_rotate_normalization;
	Eigen::RowVector2f transform_normalization;
	Eigen::Matrix2f scale_rotate_unnormalization;
	Eigen::RowVector2f transform_unnormalization;

	Eigen::Matrix2f scale_rotate_from_mean;
	Eigen::RowVector2f transform_from_mean;

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
	std::vector<Eigen::MatrixX2f> residual_model;
};

void cut_name(std::string &name, const std::string &name_with_info);

void getfiles(std::vector<std::string> &names, const std::string &path);

bool IsDetected(const cv::Rect &box, const float &x_max, const float &x_min, const float &y_max, const float &y_min);

void Loadimages(std::vector<sample> &data, const std::string &path);

void output(const sample &data, const std::string &path);

void GenerateTraindata(std::vector<sample> &data, const int &initialization);

void GenerateValidationdata(std::vector<sample> &data, const Eigen::MatrixX2f &global_mean_landmarks);

void compute_similarity_transform(
	const Eigen::MatrixX2f& target,
	const Eigen::MatrixX2f& origin,
	Eigen::Matrix2f& scale_rotate,
	Eigen::RowVector2f& transform);

void normalization(
	Eigen::MatrixX2f &target,
	const Eigen::MatrixX2f &origin,
	const Eigen::Matrix2f &scale_rotate,
	const Eigen::RowVector2f &transform);

void check_edge(sample &data);

float compute_Error(const std::vector<sample> &data);

#endif