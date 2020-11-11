#pragma once
#ifndef _ERT_HPP_
#define _ERT_HPP_

// #include <boost/property_tree/ptree.hpp>
// #include <boost/property_tree/xml_parser.hpp>
// #include <boost/foreach.hpp>
#include <string>
#include <set>
#include <exception>
#include <iostream>

#include <regressor.hpp>

namespace ert {

class ERT{
public:
	ERT(
		const int &cascade_number = 10,
		const int &tree_number = 500,
		const int &multiple_trees_number = 10,
		const int &tree_depth = 5,
		const int &feature_number_of_node = 20, 
		const int &feature_pool_size = 400,
		const float &shrinkage_factor = 0.1,
		const float &padding = 0.1,
		const int &initialization = 1,
		const float &lambda = 0.1
		);

	void train(std::vector<Sample> &data, std::vector<Sample> &validationdata, const std::string& output_path);
	void save(const std::string &path) const;
	void save_binary(const std::string& path) const;

	void load(const std::string& path);
	void load_binary(const std::string& path);

	int get_root_number() const { return (int)std::pow(2, tree_depth - 1) - 1; }
	int get_leaf_number() const { return (int)std::pow(2, tree_depth - 1); }
	int get_landmark_number() const { return (int)global_mean_landmarks.rows(); }

	void find_landmark(const cv::Mat_<uchar> image, const Eigen::Vector4f& face_rect, Eigen::MatrixX2f& landmark) const;
	void find_landmark(const cv::Mat_<uchar> image, Eigen::MatrixX2f& landmark) const;

private:
	void compute_mean_landmarks(const std::vector<Sample> &data);

private:
	int feature_number_of_node;
	int feature_pool_size;

	int cascade_number;
	int tree_number;
	int multiple_trees_number;
	int tree_depth;
	float padding;
	float shrinkage_factor;

	int initialization;
	float lambda;
	
	Eigen::MatrixX2f global_mean_landmarks;
	std::vector<Regressor> regressors;
};

} // namespace ert

#endif