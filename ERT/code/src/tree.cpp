#include <tree.hpp>
#include <Eigen/Dense>

using namespace ert;

Tree::Tree(const int &depth, const int &feature_number_of_node, const float &lambda)
{
	this->depth = depth;
	this->feature_number_of_node = feature_number_of_node;
	this->lambda = lambda;

	this->_root_number = (int)std::pow(2, depth - 1) - 1;
	this->_leaf_number = (int)std::pow(2, depth - 1);

	_model.split_model.resize(_root_number);
	_model.residual_model.resize(_leaf_number);
}

void Tree::generate_candidate_feature(
	const Eigen::MatrixX2f &feature_pool,
	const Eigen::MatrixX2f &offset,
	const std::vector<int> &landmark_index, 
    Eigen::MatrixX2f &candidate_feature_offset,
	std::vector<int> &candidate_landmark_index,
	std::vector<float> &threshold)
{
	 for(int i = 0; i < feature_number_of_node / 2; ++i)
	 {
	 	int _x_index;
	 	int _y_index;
	 	float prob_threshold;
	 	float prob;
	 	float distance;

	 	do {
	 		_x_index = std::rand() % landmark_index.size();
	 		_y_index = std::rand() % landmark_index.size();
	 		distance = (feature_pool.row(_x_index) - feature_pool.row(_y_index)).squaredNorm();
	 		prob = std::exp(-distance / lambda);
	 		prob_threshold = std::rand() / (float)(RAND_MAX);
	 	} while (_x_index == _y_index || prob <= prob_threshold);

	 	candidate_landmark_index[2*i  ] = landmark_index[_x_index];
	 	candidate_landmark_index[2*i+1] = landmark_index[_y_index];
	 	candidate_feature_offset.row(2*i  ) = offset.row(_x_index);
	 	candidate_feature_offset.row(2*i+1) = offset.row(_y_index);

	 	threshold[i] = (float)(((std::rand() / (RAND_MAX + 1.0) * std::numeric_limits<uchar>::max()) - 128) / 2.0);
	 }
}

float Tree::split_node(
	std::vector<Sample> &data,
	const Node& node,
	int index,
	bool whether_change_index)
{
	float score_left = 0;
	float score_right = 0;
	int landmark_number = (int)data[0].landmarks_cur_normalization.rows();
	Eigen::MatrixX2f mean_left(landmark_number, 2);
	Eigen::MatrixX2f mean_right(landmark_number, 2);
	mean_left.setZero();
	mean_right.setZero();
	int left_number = 0;
	int right_number = 0;

	for (int i = 0; i < data.size(); ++i)
	{
		if (data[i].tree_index == index)
		{
			bool left_node = node.evaluate(
				data[i].image,
				data[i].landmarks_cur_normalization,
				data[i].scale_rotate_from_mean,
				data[i].scale_rotate_unnormalization,
				data[i].transform_unnormalization
				);

			data[i].tree_index = 2 * index + (left_node ? 1 : 2);

			if (!whether_change_index)
			{
				if (data[i].tree_index == 2 * index + 1)
				{
					mean_left += (data[i].landmarks_truth_normalization - data[i].landmarks_cur_normalization);
					++left_number;	
				}
				else if (data[i].tree_index == 2 * index + 2)
				{
					mean_right += (data[i].landmarks_truth_normalization - data[i].landmarks_cur_normalization);
					++right_number;
				}
				data[i].tree_index = index;
			}
		}
	}
	if(!whether_change_index)
	{
		mean_left /= (float)left_number;
		mean_right /= (float)right_number;
		score_left = left_number * mean_left.squaredNorm();
		score_right = right_number * mean_right.squaredNorm();

		return score_left + score_right;
	}
	return -1;
}

void Tree::train(std::vector<Sample> &data, std::vector<Sample> &validationdata, const Eigen::MatrixX2f &feature_pool, const Eigen::MatrixX2f &offset, const std::vector<int> &landmark_index)
{
	for(int i = 0; i < _root_number; ++i)
	{
		Eigen::MatrixX2f candidate_feature_offset(feature_number_of_node, 2);
		std::vector<int> candidate_landmark_index(feature_number_of_node);
		std::vector<float> threshold(feature_number_of_node / 2);
		Tree::generate_candidate_feature(feature_pool, offset, landmark_index, candidate_feature_offset, candidate_landmark_index, threshold);

		float max_score = 0.0f;
		int index_max_score = -1;
		
		for(int j = 0; j < feature_number_of_node / 2; ++j)
		{
			float score = Tree::split_node(
				data,
				Node(
					candidate_landmark_index[2*j  ],
					candidate_landmark_index[2*j+1],
					candidate_feature_offset.row(2*j  ),
					candidate_feature_offset.row(2*j+1),
					threshold[j]),
				i, false);
			
			if(j == 0 || max_score < score)
			{
				max_score = score;
				index_max_score = 2 * j;
			}
		}
		
		_model.split_model[i] = Node(
			candidate_landmark_index[index_max_score],
			candidate_landmark_index[index_max_score + 1],
			candidate_feature_offset.row(index_max_score),
			candidate_feature_offset.row(index_max_score + 1),
			threshold[index_max_score / 2]);
		
		Tree::split_node(data, _model.split_model[i], i, true);
		Tree::split_node(validationdata, _model.split_model[i], i, true);
	}
	int landmark_number = (int)data[0].landmarks_truth_normalization.rows();
	for(int i = 0; i < _leaf_number; ++i)
	{
		_model.residual_model[i].resize(landmark_number, 2);
		_model.residual_model[i].setZero();
	}
	
	std::vector<int> data_number;
	data_number.resize(_leaf_number);
	memset(data_number.data(), 0, _leaf_number * sizeof(int));
	
	for(int i = 0; i < data.size(); ++i)
	{
		int leaf = data[i].tree_index - _root_number;
		_model.residual_model[leaf] += (data[i].landmarks_truth_normalization - data[i].landmarks_cur_normalization);
		++data_number[leaf];
	}

	for(int i = 0; i < _leaf_number; ++i)
	{
		if(data_number[i])
			_model.residual_model[i] /= (float)data_number[i];
	}
}
