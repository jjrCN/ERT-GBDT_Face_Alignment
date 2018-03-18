#include <regressor.hpp>

regressor::regressor(const int &tree_number, const int &multiple_trees_number, const int &tree_depth, 
		const int &feature_number_of_node, const int &feature_pool_size, const float &shrinkage_factor, const float &padding, const float &lamda)
{	
	this->tree_number = tree_number;
	this->multiple_trees_number = multiple_trees_number;
	this->tree_depth = tree_depth;
	this->feature_number_of_node = feature_number_of_node;
	this->feature_pool_size = feature_pool_size;
	this->shrinkage_factor = shrinkage_factor;
	this->padding = padding;
	this->lamda = lamda;

	feature_pool = cv::Mat_<float>::zeros(feature_pool_size, 2);
	offset = cv::Mat_<float>::zeros(feature_pool_size, 2);
	landmark_index.resize(feature_pool_size);

	tree tree_template(tree_depth, feature_number_of_node, lamda);

	for(int i = 0; i < tree_number; ++i)
	{
		this->_trees.push_back(tree_template);
	}
}

void regressor::compute_similarity_transform_with_mean(std::vector<sample> &data, const cv::Mat_<float> &global_mean_landmarks)
{
	cv::Mat_<float> scale_rotate(2, 2);
	cv::Mat_<float> transform(1, 2);
	for(int i = 0; i < data.size(); ++i)
	{
		compute_similarity_transform(data[i].landmarks_cur_normalization, global_mean_landmarks, scale_rotate, transform);
		data[i].scale_rotate_from_mean = scale_rotate.clone();
		data[i].transform_from_mean = transform.clone();
	}
}

void regressor::generate_feature_pool(const cv::Mat_<float> &global_mean_landmarks)
{
	float x_min, x_max, y_min, y_max;
	for(int i = 0; i < global_mean_landmarks.rows; ++i)
	{
		if(i == 0)
		{
			x_min = global_mean_landmarks(0, 0);
			x_max = global_mean_landmarks(0, 0);
			y_min = global_mean_landmarks(0, 1);
			y_max = global_mean_landmarks(0, 1);
		}
		else
		{
			if(x_min > global_mean_landmarks(i, 0))
				x_min = global_mean_landmarks(i, 0);
			if(x_max < global_mean_landmarks(i, 0))
				x_max = global_mean_landmarks(i, 0);
			if(y_min > global_mean_landmarks(i, 1))
				y_min = global_mean_landmarks(i, 1);
			if(y_max < global_mean_landmarks(i, 1))
				y_max = global_mean_landmarks(i, 1);
		}
	}

	x_min -= padding; x_max += padding;
	y_min -= padding; y_max += padding;

	for(int i = 0; i < feature_pool_size; ++i)
	{
		feature_pool(i, 0) = (std::rand() / (float)(RAND_MAX)) * (x_max - x_min) + x_min;
		feature_pool(i, 1) = (std::rand() / (float)(RAND_MAX)) * (y_max - y_min) + y_min;

		float min_distance;
		float distance;
		int index;
		for(int j = 0; j < global_mean_landmarks.rows; ++j)
		{	
			distance = std::pow((feature_pool(i, 0) - global_mean_landmarks(j, 0)), 2) + std::pow((feature_pool(i, 1) - global_mean_landmarks(j, 1)), 2);
			if(j == 0)
			{
				min_distance = distance;
				index = 0;
			}
			else
			{
				if(min_distance > distance)
				{
					min_distance = distance;
					index = j;
				}
			}
		}
		landmark_index[i] = index;
		offset(i, 0) = feature_pool(i, 0) - global_mean_landmarks(index, 0);
		offset(i, 1) = feature_pool(i, 1) - global_mean_landmarks(index, 1);
	}
}

void regressor::show_feature_node(const sample &data)
{
	cv::Mat_<uchar> image = data.image;
	cv::Mat_<float> node(feature_pool_size, 2);
	cv::Mat_<float> temp(feature_pool_size, 2);

	normalization(node, feature_pool, data.scale_rotate_from_mean, data.transform_from_mean);
	normalization(temp, node, data.scale_rotate_unnormalization, data.transform_unnormalization);

	for(int i = 0; i < feature_pool_size; ++i)
	{	
		
		float x = temp(i, 0);
		float y = temp(i, 1);
		cv::circle(image, cv::Point(x, y), 3, cv::Scalar(100, 100, 100), -1);
	}

	std::string path = "./../feature/" + data.image_name + ".jpg";
	cv::imwrite(path.c_str(), image);
}

void regressor::train(std::vector<sample> &data, std::vector<sample> &validationdata, const cv::Mat_<float> &global_mean_landmarks)
{
	regressor::compute_similarity_transform_with_mean(data, global_mean_landmarks);
	regressor::compute_similarity_transform_with_mean(validationdata, global_mean_landmarks);
	regressor::generate_feature_pool(global_mean_landmarks);

	int landmark_number = global_mean_landmarks.rows;
	int time = tree_number / multiple_trees_number;
	for(int i = 0; i < time; ++i)
	{	

		std::vector<std::vector<int>> result;
		std::vector<std::vector<int>> result_vali;
		for(int j = 0; j < multiple_trees_number; ++j)
		{

			std::vector<int> temp;
			temp.resize(data.size());
			std::vector<int> temp_vali;
			temp_vali.resize(validationdata.size());

			_trees[i * multiple_trees_number + j].train(data, validationdata, feature_pool, offset, landmark_index);

			for(int k = 0; k < data.size(); ++k)
			{
				temp[k] = data[k].tree_index - _trees[i * multiple_trees_number + j].root_number();
				data[k].tree_index = 0;
			}
			for(int k = 0; k < validationdata.size(); ++k)
			{
				temp_vali[k] = validationdata[k].tree_index - _trees[i * multiple_trees_number + j].root_number();
				validationdata[k].tree_index = 0;
			}
			result.push_back(temp);
			result_vali.push_back(temp_vali);
		}

		for(int k = 0; k < data.size(); ++k)
		{
			cv::Mat_<float> residual = cv::Mat_<float>::zeros(landmark_number, 2);
			for(int j = 0; j < multiple_trees_number; ++j)
			{
				residual += _trees[i * multiple_trees_number + j].model()->residual_model[result[j][k]];
			}
			residual /= multiple_trees_number;
			data[k].landmarks_cur_normalization += shrinkage_factor * residual;
			normalization(data[k].landmarks_cur, data[k].landmarks_cur_normalization, data[k].scale_rotate_unnormalization, data[k].transform_unnormalization);	
		}

		for(int k = 0; k < validationdata.size(); ++k)
		{
			cv::Mat_<float> residual = cv::Mat_<float>::zeros(landmark_number, 2);

			for(int j = 0; j < multiple_trees_number; ++j)
			{
				residual += _trees[i * multiple_trees_number + j].model()->residual_model[result_vali[j][k]];
			}
			residual /= multiple_trees_number;
			validationdata[k].landmarks_cur_normalization += shrinkage_factor * residual;
			normalization(validationdata[k].landmarks_cur, validationdata[k].landmarks_cur_normalization, 
							validationdata[k].scale_rotate_unnormalization, validationdata[k].transform_unnormalization);
		}
		if(time < 10)
		{
			std::cout << (i + 1) * multiple_trees_number << "\ttrees completed." << std::endl;
		}
		else
		{
			if(i % (time / 10) == 0)
			{
				std::cout << (i / (time / 10)) * 10 << "%\ttrees completed." << std::endl;
			}
		}
	}
}