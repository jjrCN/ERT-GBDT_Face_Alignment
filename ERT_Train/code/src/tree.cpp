#include <tree.hpp>

tree::tree(const int &depth, const int &feature_number_of_node, const float &lamda)
{
	this->depth = depth;
	this->feature_number_of_node = feature_number_of_node;
	this->lamda = lamda;

	this->_root_number = std::pow(2, depth - 1) - 1;
	this->_leaf_number = std::pow(2, depth - 1);

	_model.splite_model.resize(_root_number);
	_model.residual_model.resize(_leaf_number);
}

void tree::generate_candidate_feature(const cv::Mat_<float> &feature_pool, const cv::Mat_<float> &offset, const std::vector<int> &landmark_index, 
										cv::Mat_<float> &candidate_feature_offset, std::vector<int> &candidate_landmark_index, std::vector<float> &threshold)
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

	 	threshold[i] = ((std::rand() / (RAND_MAX + 1.0) * std::numeric_limits<uchar>::max()) -128) / 2;
	 }
}

float tree::splite_node(std::vector<sample> &data, const float &u_x, const float &u_y, const float &v_x, const float &v_y, const int &u_index, 
						const int &v_index, const float &threshold, const int &index, bool whether_change_index)
{
	float score_left = 0;
	float score_right = 0;
	int landmark_number = data[0].landmarks_cur_normalization.rows;
	cv::Mat_<float> mean_left = cv::Mat_<float>::zeros(landmark_number, 2);
	cv::Mat_<float> mean_right = cv::Mat_<float>::zeros(landmark_number, 2);
	int left_number = 0;
	int right_number = 0;
	for(int i = 0; i < data.size(); ++i)
	{
		if(data[i].tree_index == index)
		{
			cv::Mat_<float> u_offset_temp(1, 2);
			u_offset_temp(0, 0) = u_x;
			u_offset_temp(0, 1) = u_y;

			cv::Mat_<float> v_offset_temp(1, 2);
			v_offset_temp(0, 0) = v_x;
			v_offset_temp(0, 1) = v_y;

			cv::Mat_<float> u_data(1, 2);
			cv::Mat_<float> v_data(1, 2);
			cv::Mat_<float> u_cur(1, 2);
			cv::Mat_<float> v_cur(1, 2);
			u_cur(0, 0) = data[i].landmarks_cur_normalization(u_index, 0); u_cur(0, 1) = data[i].landmarks_cur_normalization(u_index, 1);
			v_cur(0, 0) = data[i].landmarks_cur_normalization(v_index, 0); v_cur(0, 1) = data[i].landmarks_cur_normalization(v_index, 1);

			u_data = u_cur + u_offset_temp * data[i].scale_rotate_from_mean;
			v_data = v_cur + v_offset_temp * data[i].scale_rotate_from_mean;
			
			cv::Mat_<float> u_data_unnormalization(1, 2);
			cv::Mat_<float> v_data_unnormalization(1, 2);

			normalization(u_data_unnormalization, u_data, data[i].scale_rotate_unnormalization, data[i].transform_unnormalization);
			normalization(v_data_unnormalization, v_data, data[i].scale_rotate_unnormalization, data[i].transform_unnormalization);
			int u_value, v_value;

			//std::cout << i << std::endl;
			
			if(u_data_unnormalization(0, 0) < 0 || u_data_unnormalization(0, 0) > data[i].image.cols || 
				u_data_unnormalization(0, 1) < 0 || u_data_unnormalization(0, 1) > data[i].image.rows)
				u_value = 0;
			else
				u_value = data[i].image.at<uchar>((int)u_data_unnormalization(0, 1), (int)u_data_unnormalization(0, 0));

			if(v_data_unnormalization(0, 0) < 0 || v_data_unnormalization(0, 0) > data[i].image.cols || 
				v_data_unnormalization(0, 1) < 0 || v_data_unnormalization(0, 1) > data[i].image.rows)
				v_value = 0;
			else
				v_value = data[i].image.at<uchar>((int)v_data_unnormalization(0, 1), (int)v_data_unnormalization(0, 0));

			if(u_value - v_value > threshold)
				data[i].tree_index = 2 * index + 1;
			else
				data[i].tree_index = 2 * index + 2;

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
		mean_left /= left_number;
		mean_right /= right_number;
		score_left = left_number * mean_left.dot(mean_left);
		score_right = right_number * mean_right.dot(mean_right);

		return score_left + score_right;
	}
	return -1;
}

void tree::train(std::vector<sample> &data, std::vector<sample> &validationdata, const cv::Mat_<float> &feature_pool, const cv::Mat_<float> &offset, const std::vector<int> &landmark_index)
{
	for(int i = 0; i < _root_number; ++i)
	{
		cv::Mat_<float> candidate_feature_offset(feature_number_of_node, 2);
		std::vector<int> candidate_landmark_index(feature_number_of_node);
		std::vector<float> threshold(feature_number_of_node / 2);
		tree::generate_candidate_feature(feature_pool, offset, landmark_index, candidate_feature_offset, candidate_landmark_index, threshold);
		float max_score;
		int index_max_score;
		
		for(int j = 0; j < feature_number_of_node / 2; ++j)
		{
			float score =  tree::splite_node(data, candidate_feature_offset(2 * j, 0), candidate_feature_offset(2 * j, 1), candidate_feature_offset(2 * j + 1, 0), 
				candidate_feature_offset(2 * j + 1, 1), candidate_landmark_index[2 * j], candidate_landmark_index[2 * j + 1], threshold[j], i, false);
			
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
		_model.splite_model[i].index1_offset_x = candidate_feature_offset(index_max_score, 0);
		_model.splite_model[i].index1_offset_y = candidate_feature_offset(index_max_score, 1);
		_model.splite_model[i].index2_offset_x = candidate_feature_offset(index_max_score + 1, 0);
		_model.splite_model[i].index2_offset_y = candidate_feature_offset(index_max_score + 1, 1);
		_model.splite_model[i].threshold = threshold[index_max_score / 2];
		
		tree::splite_node(data, _model.splite_model[i].index1_offset_x, _model.splite_model[i].index1_offset_y, _model.splite_model[i].index2_offset_x,
			 _model.splite_model[i].index2_offset_y, _model.splite_model[i].landmark_index1, _model.splite_model[i].landmark_index2, _model.splite_model[i].threshold, i, true);

		tree::splite_node(validationdata, _model.splite_model[i].index1_offset_x, _model.splite_model[i].index1_offset_y, _model.splite_model[i].index2_offset_x,
			 _model.splite_model[i].index2_offset_y, _model.splite_model[i].landmark_index1, _model.splite_model[i].landmark_index2, _model.splite_model[i].threshold, i, true);

	}
	int landmark_number = data[0].landmarks_truth_normalizaiotn.rows;
	for(int i = 0; i < _leaf_number; ++i)
	{
		_model.residual_model[i] = cv::Mat_<float>::zeros(landmark_number, 2);
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
			_model.residual_model[i] /= data_number[i];
	}
}
