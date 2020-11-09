#pragma once
#ifndef _TREE_HPP_
#define _TREE_HPP_

#include <utils.hpp>

namespace ert {

class Tree {
public:
	Tree(const int &depth = 5, const int &feature_number_of_node = 20, const float &lambda = 0.1);

	void train(
		std::vector<Sample> &data,
		std::vector<Sample> &validationdata,
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

	float split_node(
		std::vector<Sample> &data,
		const Node& node,
		int index,
		bool whether_change_index
		);

private:
	int depth;
	int feature_number_of_node;
	float lambda;

	int _root_number;
	int _leaf_number;

	TreeModel _model;
};

} // namespace ert

#endif