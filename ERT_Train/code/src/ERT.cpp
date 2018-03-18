#include <ERT.hpp>

ERT::ERT(const int &cascade_number, const int &tree_number, const int &multiple_trees_number, const int &tree_depth, 
		const int &feature_number_of_node, const int &feature_pool_size, const float &shrinkage_factor, const float &padding, const int &initialization, const float &lamda)
{
		if(tree_number % multiple_trees_number != 0)
	{
		perror("tree_number must multiple times of multiple_trees_number.");
		exit(1);
	}
	this->cascade_number = cascade_number;
	this->tree_number = tree_number;
	this->multiple_trees_number = multiple_trees_number;
	this->tree_depth = tree_depth;
	this->feature_number_of_node = feature_number_of_node;
	this->feature_pool_size = feature_pool_size;
	this->shrinkage_factor = shrinkage_factor;
	this->padding = padding;
	this->initialization = initialization;
	this->lamda = lamda;

	regressor regressor_template(tree_number, multiple_trees_number, tree_depth, feature_number_of_node, feature_pool_size, shrinkage_factor, padding, lamda);

	for(int i = 0; i < cascade_number; ++i)
	{
		this->regressors.push_back(regressor_template);
	}
	std::cout << "ERT model has been created." << std::endl;
}

void ERT::compute_mean_landmarks(const std::vector<sample> &data)
{
	global_mean_landmarks = cv::Mat_<float>::zeros(data[0].landmarks_truth.rows, 2);
	for(int i = 0; i < data.size() / initialization; ++i)
	{
		global_mean_landmarks += data[i].landmarks_truth_normalizaiotn;
	}
	global_mean_landmarks /= (data.size() / initialization);
	std::cout << "Compute global mean landmarks finished." << std::endl << std::endl;

}

void ERT::train(std::vector<sample> &data, std::vector<sample> &validationdata)
{
	if(data.empty())
	{
		perror("input data should not be empty.");
		exit(1);
	}

	ERT::compute_mean_landmarks(data);

	GenerateValidationdata(validationdata, global_mean_landmarks);

	for(int i = 0; i < cascade_number; ++i)
	{

		if(i == 0)
		{
			std::string path = "./../train_origin_landmark";
			rmdir(path.c_str());
			mkdir(path.c_str(), S_IRWXU);
			for(int j = 0; j < 10; ++j)
				output(data[j], path);

			path = "./../validation_origin_landmark";
			rmdir(path.c_str());
			mkdir(path.c_str(), S_IRWXU);
			for(int j = 0; j < 10; ++j)
				output(validationdata[j], path);			
		}
		std::cout << "[Cascade " << i + 1 << "] Training..." << std::endl;
		regressors[i].train(data, validationdata ,global_mean_landmarks);

		std::stringstream stream;
		stream << i + 1;
		std::string outputpath_train = "./../train_cascade_" + stream.str();
		std::string outputpath_vali = "./../validation_cascade_" + stream.str();

		rmdir(outputpath_train.c_str());
		rmdir(outputpath_vali.c_str());
		mkdir(outputpath_train.c_str(), S_IRWXU);
		mkdir(outputpath_vali.c_str(), S_IRWXU);

		for(int j = 0; j < 10; ++j)
		{
			output(data[j], outputpath_train);
			output(validationdata[j], outputpath_vali);
		}

		std::cout << "training error = " << compute_Error(data) << std::endl;
		std::cout << "validation error = " << compute_Error(validationdata) << std::endl << std::endl;
	}
	std::cout << "[Finish]" << std::endl;
}

void ERT::save(const std::string &path)
{
	int root_number = std::pow(2, tree_depth - 1) - 1;
	int leaf_number = std::pow(2, tree_depth - 1);
	int landmark_number = global_mean_landmarks.rows;

	pt::ptree tree;

	tree.put("ERT.model_name", "ERT by jinrang jia");
	tree.put("ERT.cascade_number", cascade_number);
	tree.put("ERT.tree_number", tree_number);
	tree.put("ERT.multiple_trees_number", multiple_trees_number);
	tree.put("ERT.tree_depth", tree_depth);
	tree.put("ERT.landmark_number", landmark_number);
	tree.put("ERT.feature_number_of_node", feature_number_of_node);
	tree.put("ERT.feature_pool_size", feature_pool_size);
	tree.put("ERT.shrinkage_factor", shrinkage_factor);
	tree.put("ERT.padding", padding);
	tree.put("ERT.lamda", lamda);

	for(int i = 0; i < landmark_number; ++i)
	{
		std::stringstream stream_global_mean_landmarks;
		stream_global_mean_landmarks << i;
		std::string global_mean_landmarks_path_x = "ERT.parameters.global_mean_landmarks.x_" + stream_global_mean_landmarks.str();
		std::string global_mean_landmarks_path_y = "ERT.parameters.global_mean_landmarks.y_" + stream_global_mean_landmarks.str();

		tree.put(global_mean_landmarks_path_x, global_mean_landmarks(i, 0));
		tree.put(global_mean_landmarks_path_y, global_mean_landmarks(i, 1));
	}

	for(int i = 0; i < cascade_number; ++i)
	{
		std::stringstream stream_regressor;
		stream_regressor << i;
		std::string regressor = "ERT.parameters.regressor_" + stream_regressor.str() + ".";

		for(int j = 0; j < tree_number; ++j)
		{
			std::stringstream stream_tree;
			stream_tree << j;
			std::string tree_path = regressor + "tree_" + stream_tree.str() + ".";

			for(int k = 0; k < root_number; ++k)
			{
				std::stringstream stream_root_node;
				stream_root_node << k;
				std::string root_node = tree_path + "root_node_" + stream_root_node.str() + ".";

			
				std::string landmark_index1_path = root_node + "landmark_index1";
				tree.put(landmark_index1_path, regressors[i].trees()[j].model()->splite_model[k].landmark_index1);

				std::string landmark_index2_path = root_node + "landmark_index2";
				tree.put(landmark_index2_path, regressors[i].trees()[j].model()->splite_model[k].landmark_index2);

				std::string index1_offset_x = root_node + "index1_offset_x";
				tree.put(index1_offset_x, regressors[i].trees()[j].model()->splite_model[k].index1_offset_x);

				std::string index1_offset_y = root_node + "index1_offset_y";
				tree.put(index1_offset_y, regressors[i].trees()[j].model()->splite_model[k].index1_offset_y);

				std::string index2_offset_x = root_node + "index2_offset_x";
				tree.put(index2_offset_x, regressors[i].trees()[j].model()->splite_model[k].index2_offset_x);

				std::string index2_offset_y = root_node + "index2_offset_y";
				tree.put(index2_offset_y, regressors[i].trees()[j].model()->splite_model[k].index2_offset_y);

				std::string threshold = root_node + "threshold";
				tree.put(threshold, regressors[i].trees()[j].model()->splite_model[k].threshold);
			}

			for(int k = 0; k < leaf_number; ++k)
			{
				std::stringstream stream_root_node;
				stream_root_node << k;
				std::string root_node = tree_path + "leaf_node_" + stream_root_node.str() + ".";

				for(int r = 0; r < landmark_number; ++r)
				{
					std::stringstream landmark_index;
					landmark_index << r;
					std::string residual_x = root_node + "x_" + landmark_index.str();
					std::string residual_y = root_node + "y_" + landmark_index.str();

					tree.put(residual_x, regressors[i].trees()[j].model()->residual_model[k](r, 0));
					tree.put(residual_y, regressors[i].trees()[j].model()->residual_model[k](r, 1));
				}
			}
		}
	}

	remove(path.c_str());
	pt::write_xml(path, tree);
}