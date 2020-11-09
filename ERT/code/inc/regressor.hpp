#ifndef _REGRESSOR_HPP_
#define _REGRESSOR_HPP_

#include <tree.hpp>
#include <Eigen/Dense>

namespace ert {

class Regressor{
public:
	Regressor(
		const int &tree_number = 500,
		const int &multiple_trees_number = 10,
		const int &tree_depth = 5,
		const int &feature_number_of_node = 20,
		const int &feature_pool_size = 400,
		const float &shrinkage_factor = 0.1,
		const float &padding = 0.1,
		const float &lamda = 0.1
		);

	void train(std::vector<Sample> &data, std::vector<Sample> &validationdata, const Eigen::MatrixX2f &global_mean_landmarks);

	const std::vector<Tree>& trees() const {return _trees;};

private:

	void compute_similarity_transform_with_mean(std::vector<Sample> &data, const Eigen::MatrixX2f &global_mean_landmarks);

	void generate_feature_pool(const Eigen::MatrixX2f &global_mean_landmarks);

	void show_feature_node(const Sample &data);

private:
	int feature_number_of_node;
	int feature_pool_size;
	int multiple_trees_number;
	float padding;
	float shrinkage_factor;

	int tree_number;
	int tree_depth;
	float lamda;

	std::vector<Tree> _trees;

	Eigen::MatrixX2f feature_pool;
	std::vector<int> landmark_index;
	Eigen::MatrixX2f offset;
};

} // namespace ert

#endif