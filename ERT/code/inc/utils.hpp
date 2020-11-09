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

class Node{
public:
	int landmark_index1 { 0 };
	int landmark_index2 { 0 };
	Eigen::RowVector2f index1_offset { Eigen::RowVector2f::Zero() };
	Eigen::RowVector2f index2_offset { Eigen::RowVector2f::Zero() };
	float threshold { 0.0f };

	Node() {}
	Node(
		int index1,
		int index2,
		const Eigen::RowVector2f& offset1,
		const Eigen::RowVector2f& offset2,
		float _threshold)
	: landmark_index1(index1)
	, landmark_index2(index2)
	, index1_offset(offset1)
	, index2_offset(offset2)
	, threshold(_threshold) {}

	bool evaluate(
		const cv::Mat_<uchar>& image,
		const Eigen::MatrixX2f& current_normalized_shape,
		const Eigen::Matrix2f& transform_mean_to_normal,
		const Eigen::Matrix2f& transform_normal_to_image,
		const Eigen::RowVector2f& translation_normal_to_image
		) const;
};

class TreeModel{
public:
	std::vector<Node> splite_model;
	std::vector<Eigen::MatrixX2f> residual_model;
};

void load_samples(std::vector<Sample> &data, const std::string &path);

void load_pts(const std::string& filename, Eigen::MatrixX2f& points);

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