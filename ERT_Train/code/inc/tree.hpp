#pragma once
#ifndef _TREE_HPP_
#define _TREE_HPP_

#include <utilis.hpp>

class tree{
public:
	tree(const int &depth = 5, const int &feature_number_of_node = 20, const float &lamda = 0.1);

	void train(std::vector<sample> &data, std::vector<sample> &validationdata, const cv::Mat_<float> &feature_pool, const cv::Mat_<float> &offset, const std::vector<int> &landmark_index);

	int root_number(){return _root_number;};

	int leaf_number(){return _leaf_number;};

	const TreeModel* model() const {return &_model;};

private:
	void generate_candidate_feature(const cv::Mat_<float> &feature_pool, const cv::Mat_<float> &offset, const std::vector<int> &landmark_index, 
										cv::Mat_<float> &candidate_feature_offset, std::vector<int> &candidate_landmark_index, std::vector<float> &threshold);

	float splite_node(std::vector<sample> &data, const float &u_x, const float &u_y, const float &v_x, const float &v_y, 
									const int &u_index, const int &v_index, const float &threshold, const int &index, bool whether_change_index);

private:
	int depth;
	int feature_number_of_node;
	float lamda;

	int _root_number;
	int _leaf_number;

	TreeModel _model;
};

#endif