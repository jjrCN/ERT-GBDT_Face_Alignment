#pragma once
#ifndef _TREE_HPP_
#define _TREE_HPP_

#include <utilis.hpp>

class tree{
public:
	tree(const int &depth = 5, const int &feature_number_of_node = 20, const float &lamda = 0.1);

	void train(
		std::vector<sample> &data,
		std::vector<sample> &validationdata,
		const Eigen::MatrixX2f &feature_pool,
		const Eigen::MatrixX2f &offset,
		const std::vector<int> &landmark_index
		);

	int root_number() const {return _root_number;};
	int leaf_number() const {return _leaf_number;};

	const TreeModel* model() const {return &_model;};

private:
	void generate_candidate_feature(
		const Eigen::MatrixX2f &feature_pool,
		const Eigen::MatrixX2f &offset,
		const std::vector<int> &landmark_index,
		Eigen::MatrixX2f &candidate_feature_offset,
		std::vector<int> &candidate_landmark_index,
		std::vector<float> &threshold
		);

	float splite_node(
		std::vector<sample> &data,
		const float &u_x, const float &u_y,
		const float &v_x, const float &v_y, 
		const int &u_index, const int &v_index,
		const float &threshold, const int &index,
		bool whether_change_index
		);

private:
	int depth;
	int feature_number_of_node;
	float lamda;

	int _root_number;
	int _leaf_number;

	TreeModel _model;
};

#endif