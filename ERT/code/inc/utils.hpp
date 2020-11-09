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

namespace ert {

class Sample {
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
	Eigen::RowVector2f index1_offset;
	Eigen::RowVector2f index2_offset;
	float threshold;
};

class TreeModel{
public:
	std::vector<UnLeafNode> splite_model;
	std::vector<Eigen::MatrixX2f> residual_model;
};

void load_samples(std::vector<Sample> &data, const std::string &path);

void output(const Sample &data, const std::string &path);

void generate_train_data(std::vector<Sample> &data, const int &initialization);

void generate_validation_data(std::vector<Sample> &data, const Eigen::MatrixX2f &global_mean_landmarks);

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

void check_edge(Sample &data);

float compute_error(const std::vector<Sample> &data);

} // namespace ert

#endif