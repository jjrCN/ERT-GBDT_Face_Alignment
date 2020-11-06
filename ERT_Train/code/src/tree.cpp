#include <tree.hpp>
#include <Eigen/Dense>

tree::tree(const int &depth, const int &feature_number_of_node, const float &lamda)
{
	this->depth = depth;
	this->feature_number_of_node = feature_number_of_node;
	this->lamda = lamda;

	this->_root_number = (int)std::pow(2, depth - 1) - 1;
	this->_leaf_number = (int)std::pow(2, depth - 1);

	_model.splite_model.resize(_root_number);
	_model.residual_model.resize(_leaf_number);
}

void tree::generate_candidate_feature(
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

	 	do{
	 		_x_index = std::rand() % landmark_index.size();
	 		_y_index = std::rand() % landmark_index.size();
	 		distance = std::pow(feature_pool(_x_index, 0) - feature_pool(_y_index, 0), 2) + std::pow(feature_pool(_x_index, 1) - feature_pool(_y_index, 1), 2);
	 		prob = std::exp(-distance / lamda);
	 		prob_threshold = std::rand() / (float)(RAND_MAX);
	 	}while(_x_index == _y_index || prob <= prob_threshold);

	 	candidate_landmark_index[2 * i] = landmark_index[_x_index];
	 	candidate_landmark_index[2 * i + 1] = landmark_index[_y_index];

	 	candidate_feature_offset(2 * i, 0) = offset(_x_index, 0);
	 	candidate_feature_offset(2 * i, 1) = offset(_x_index, 1);
	 	candidate_feature_offset(2 * i + 1, 0) = offset(_y_index, 0);
	 	candidate_feature_offset(2 * i + 1, 1) = offset(_y_index, 1);

	 	threshold[i] = (float)(((std::rand() / (RAND_MAX + 1.0) * std::numeric_limits<uchar>::max()) - 128) / 2.0);
	 }
}

float tree::splite_node(
	std::vector<sample> &data,
	const Eigen::RowVector2f &u,
	const Eigen::RowVector2f &v, 
	int u_index, int v_index,
	float threshold, int index,
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

	for(int i = 0; i < data.size(); ++i)
	{
		if(data[i].tree_index == index)
		{
			auto u_cur = data[i].landmarks_cur_normalization.row(u_index);
			auto v_cur = data[i].landmarks_cur_normalization.row(v_index);

			auto u_data = u_cur + u * data[i].scale_rotate_from_mean;
			auto v_data = v_cur + v * data[i].scale_rotate_from_mean;

			auto u_data_unnormalization = u_data * data[i].scale_rotate_unnormalization + data[i].transform_unnormalization;
			auto v_data_unnormalization = v_data * data[i].scale_rotate_unnormalization + data[i].transform_unnormalization;

			int u_value = 0;
			int v_value = 0;
			if (u_data_unnormalization(0) >= 0 && u_data_unnormalization(0) < data[i].image.cols &&
				u_data_unnormalization(1) >= 0 && u_data_unnormalization(1) < data[i].image.rows)
				u_value = data[i].image.at<uchar>((int)u_data_unnormalization(1), (int)u_data_unnormalization(0));

			if (v_data_unnormalization(0) >= 0 && v_data_unnormalization(0) < data[i].image.cols &&
				v_data_unnormalization(1) >= 0 && v_data_unnormalization(1) < data[i].image.rows)
				v_value = data[i].image.at<uchar>((int)v_data_unnormalization(1), (int)v_data_unnormalization(0));

			data[i].tree_index = 2 * index + ((u_value - v_value > threshold) ? 1 : 2);

			if(!whether_change_index)
			{
				if(data[i].tree_index == 2 * index + 1)
				{
					mean_left += (data[i].landmarks_truth_normalizaiotn - data[i].landmarks_cur_normalization);
					++left_number;	
				}
				else if(data[i].tree_index == 2 * index + 2)
				{
					mean_right += (data[i].landmarks_truth_normalizaiotn - data[i].landmarks_cur_normalization);
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

void tree::train(std::vector<sample> &data, std::vector<sample> &validationdata, const Eigen::MatrixX2f &feature_pool, const Eigen::MatrixX2f &offset, const std::vector<int> &landmark_index)
{
	for(int i = 0; i < _root_number; ++i)
	{
		Eigen::MatrixX2f candidate_feature_offset(feature_number_of_node, 2);
		std::vector<int> candidate_landmark_index(feature_number_of_node);
		std::vector<float> threshold(feature_number_of_node / 2);
		tree::generate_candidate_feature(feature_pool, offset, landmark_index, candidate_feature_offset, candidate_landmark_index, threshold);
		float max_score;
		int index_max_score;
		
		for(int j = 0; j < feature_number_of_node / 2; ++j)
		{
			float score =  tree::splite_node(data,
				candidate_feature_offset.row(2 * j),
				candidate_feature_offset.row(2 * j + 1),
				candidate_landmark_index[2 * j],
				candidate_landmark_index[2 * j + 1],
				threshold[j], i, false);
			
			if(j == 0)
			{
				max_score = score;
				index_max_score = 0;
			}
			else
			{
				if(max_score < score)
				{
					max_score = score;
					index_max_score = 2 * j;
				}
			}
		}
		
		_model.splite_model[i].landmark_index1 = candidate_landmark_index[index_max_score];
		_model.splite_model[i].landmark_index2 = candidate_landmark_index[index_max_score + 1];
		_model.splite_model[i].index1_offset = candidate_feature_offset.row(index_max_score);
		_model.splite_model[i].index2_offset = candidate_feature_offset.row(index_max_score + 1);
		_model.splite_model[i].threshold = threshold[index_max_score / 2];
		
		tree::splite_node(data,
			_model.splite_model[i].index1_offset,
			_model.splite_model[i].index2_offset,
			_model.splite_model[i].landmark_index1,
			_model.splite_model[i].landmark_index2,
			_model.splite_model[i].threshold, i, true);

		tree::splite_node(validationdata,
			_model.splite_model[i].index1_offset,
			_model.splite_model[i].index2_offset,
			_model.splite_model[i].landmark_index1,
			_model.splite_model[i].landmark_index2,
			_model.splite_model[i].threshold, i, true);

	}
	int landmark_number = (int)data[0].landmarks_truth_normalizaiotn.rows();
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
		_model.residual_model[leaf] += (data[i].landmarks_truth_normalizaiotn - data[i].landmarks_cur_normalization);
		++data_number[leaf];
	}

	for(int i = 0; i < _leaf_number; ++i)
	{
		if(data_number[i])
			_model.residual_model[i] /= (float)data_number[i];
	}
}
