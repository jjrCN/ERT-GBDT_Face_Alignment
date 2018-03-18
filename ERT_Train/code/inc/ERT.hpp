#pragma once
#ifndef _ERT_HPP_
#define _ERT_HPP_

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/foreach.hpp>
#include <string>
#include <set>
#include <exception>
#include <iostream>

#include <regressor.hpp>

namespace pt = boost::property_tree;

class ERT{
public:
	ERT(const int &cascade_number = 10, const int &tree_number = 500, const int &multiple_trees_number = 10, const int &tree_depth = 5, const int &feature_number_of_node = 20, 
		const int &feature_pool_size = 400, const float &shrinkage_factor = 0.1, const float &padding = 0.1, const int &initialization = 1, const float &lamda = 0.1);

	void train(std::vector<sample> &data, std::vector<sample> &validationdata);

	void save(const std::string &path);

private:
	void compute_mean_landmarks(const std::vector<sample> &data);

private:
	cv::Mat_<float> global_mean_landmarks;

	int feature_number_of_node;
	int feature_pool_size;

	int cascade_number;
	int tree_number;
	int multiple_trees_number;
	int tree_depth;
	float padding;
	float shrinkage_factor;

	int initialization;
	float lamda;
	
	std::vector<regressor> regressors;
};

#endif