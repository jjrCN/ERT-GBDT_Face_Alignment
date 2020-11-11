#include <ERT.hpp>
#include <ghc/filesystem.hpp>
#include <nlohmann/json.hpp>

namespace fs = ghc::filesystem;
using namespace nlohmann;
using namespace ert;

ERT::ERT(const int &cascade_number, const int &tree_number, const int &multiple_trees_number, const int &tree_depth, 
		const int &feature_number_of_node, const int &feature_pool_size, const float &shrinkage_factor, const float &padding, const int &initialization, const float &lambda)
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
	this->lambda = lambda;

	Regressor regressor_template(tree_number, multiple_trees_number, tree_depth, feature_number_of_node, feature_pool_size, shrinkage_factor, padding, lambda);

	for(int i = 0; i < cascade_number; ++i)
	{
		this->regressors.push_back(regressor_template);
	}
	std::cout << "ERT model has been created." << std::endl;
}

void ERT::compute_mean_landmarks(const std::vector<Sample> &data)
{
	global_mean_landmarks.resize(data[0].landmarks_truth.rows(), 2);
	global_mean_landmarks.setZero();
	for(int i = 0; i < data.size() / initialization; ++i)
	{
		global_mean_landmarks += data[i].landmarks_truth_normalization;
	}
	// global_mean_landmarks /= (data.size() / initialization);
	global_mean_landmarks /= ((float)data.size() / (float)initialization);
	std::cout << "Compute global mean landmarks finished." << std::endl << std::endl;

}

void ERT::train(std::vector<Sample> &data, std::vector<Sample> &validationdata, const std::string& output_path)
{
	if(data.empty())
	{
		perror("input data should not be empty.");
		exit(1);
	}

	ERT::compute_mean_landmarks(data);

	generate_validation_data(validationdata, global_mean_landmarks);

	for(int i = 0; i < cascade_number; ++i)
	{

		if(i == 0)
		{
			fs::path path = fs::path(output_path) / "train_origin_landmark";
			fs::remove_all(path);
			fs::create_directories(path);
			for(int j = 0; j < 10; ++j)
				output(data[j], path);

			path = fs::path(output_path) / "validation_origin_landmark";
			fs::remove_all(path);
			fs::create_directories(path);
			for(int j = 0; j < 10; ++j)
				output(validationdata[j], path);			
		}
		std::cout << "[Cascade " << i + 1 << "] Training..." << std::endl;
		regressors[i].train(data, validationdata ,global_mean_landmarks);

		std::stringstream stream;
		stream << i + 1;
		fs::path outputpath_train = fs::path(output_path) / ("train_cascade_" + stream.str());
		fs::path outputpath_vali  = fs::path(output_path) / ("validation_cascade_" + stream.str());

		fs::remove_all(outputpath_train);
		fs::remove_all(outputpath_vali);
		fs::create_directories(outputpath_train);
		fs::create_directories(outputpath_vali);

		for(int j = 0; j < 10; ++j)
		{
			output(data[j], outputpath_train);
			output(validationdata[j], outputpath_vali);
		}

		std::cout << "training error = " << compute_error(data) << std::endl;
		std::cout << "validation error = " << compute_error(validationdata) << std::endl << std::endl;
	}
	std::cout << "[Finish]" << std::endl;
}

void ERT::save(const std::string &path) const
{
	int root_number = get_root_number();
	int leaf_number = get_leaf_number();
	int landmark_number = get_landmark_number();

	json model;
	auto& info = model["info"];
	info["model_name"] = "ERT by jinrang jia";
	info["cascade_number"] = cascade_number;
	info["tree_number"] = tree_number;
	info["multiple_trees_number"] = multiple_trees_number;
	info["tree_depth"] = tree_depth;
	info["landmark_number"] = landmark_number;
	info["feature_number_of_node"] = feature_number_of_node;
	info["feature_pool_size"] = feature_pool_size;
	info["shrinkage_factor"] = shrinkage_factor;
	info["padding"] = padding;
	info["lambda"] = lambda;

	auto& gml_json = model["global_mean_landmark"];
	for(int i = 0; i < landmark_number; ++i)
	{
		gml_json[2*i  ] = global_mean_landmarks(i, 0);
		gml_json[2*i+1] = global_mean_landmarks(i, 1);
	}

	auto& regressor_json = model["regressor"];
	for(int i = 0; i < cascade_number; ++i)
	{
		auto& forest_json = regressor_json[i];
		for(int j = 0; j < tree_number; ++j)
		{
			auto& node_json = forest_json[j]["node"];
			for(int k = 0; k < root_number; ++k)
			{
				node_json[k]["landmark_index"][0] = regressors[i].trees()[j].model().split_model[k].landmark_index1;
				node_json[k]["landmark_index"][1] = regressors[i].trees()[j].model().split_model[k].landmark_index2;
				node_json[k]["offset"][0] = regressors[i].trees()[j].model().split_model[k].index1_offset(0);
				node_json[k]["offset"][1] = regressors[i].trees()[j].model().split_model[k].index1_offset(1);
				node_json[k]["offset"][2] = regressors[i].trees()[j].model().split_model[k].index2_offset(0);
				node_json[k]["offset"][3] = regressors[i].trees()[j].model().split_model[k].index2_offset(1);
				node_json[k]["threshold"] = regressors[i].trees()[j].model().split_model[k].threshold;
			}

			auto& leaves_json = forest_json[j]["leaf"];
			for(int k = 0; k < leaf_number; ++k)
			{
				auto& leaf_json = leaves_json[k];
				for(int r = 0; r < landmark_number; ++r)
				{
					leaf_json[2*r  ] = regressors[i].trees()[j].model().residual_model[k](r, 0);
					leaf_json[2*r+1] = regressors[i].trees()[j].model().residual_model[k](r, 1);
				}
			}
		}
	}

	fs::remove(path);
	std::ofstream fout(path);
	fout << model.dump(2);
	fout.close();
}

void ERT::save_binary(const std::string& path) const
{
	int root_number = get_root_number();
	int leaf_number = get_leaf_number();
	int landmark_number = get_landmark_number();

	std::ofstream fout(path, std::ios::binary);

	// general info
	fout.write((const char*)&cascade_number, sizeof(int));
	fout.write((const char*)&tree_number, sizeof(int));
	fout.write((const char*)&multiple_trees_number, sizeof(int));
	fout.write((const char*)&tree_depth, sizeof(int));
	fout.write((const char*)&landmark_number, sizeof(int));
	fout.write((const char*)&feature_number_of_node, sizeof(int));
	fout.write((const char*)&feature_pool_size, sizeof(int));
	fout.write((const char*)&shrinkage_factor, sizeof(float));
	fout.write((const char*)&padding, sizeof(float));
	fout.write((const char*)&lambda, sizeof(float));

	// global landmark mean
	fout.write((const char*)global_mean_landmarks.data(), sizeof(float) * landmark_number * 2);

	// regressors
	for(int i = 0; i < cascade_number; ++i)
	{
		for(int j = 0; j < tree_number; ++j)
		{
			auto& model = regressors[i].trees()[j].model();
			for(int k = 0; k < root_number; ++k)
			{
				auto& node = model.split_model[k];
				fout.write((const char*)&node, sizeof(Node));
			}

			for(int k = 0; k < leaf_number; ++k)
			{
				auto& leaf = model.residual_model[k];
				fout.write((const char*)leaf.data(), sizeof(float) * landmark_number * 2);
			}
		}
	}

	fout.close();
}

void ERT::load(const std::string& path)
{
	std::ifstream fin(path);
	json tree_json;
	fin >> tree_json;
	fin.close();

	std::cout << "open model." << std::endl;

	cascade_number = tree_json["info"]["cascade_number"].get<int>();
	tree_number = tree_json["info"]["tree_number"].get<int>();
	multiple_trees_number = tree_json["info"]["multiple_trees_number"].get<int>();
	tree_depth = tree_json["info"]["tree_depth"].get<int>();
	shrinkage_factor = tree_json["info"]["shrinkage_factor"].get<float>();

	int root_number = get_root_number();
	int leaf_number = get_leaf_number();
	int landmark_number = tree_json["info"]["landmark_number"].get<int>();

	global_mean_landmarks.resize(landmark_number, 2);
	for(int i = 0; i < landmark_number; ++i)
	{
		global_mean_landmarks(i, 0) = tree_json["global_mean_landmark"][2*i  ].get<float>();
		global_mean_landmarks(i, 1) = tree_json["global_mean_landmark"][2*i+1].get<float>();
	}

	regressors.resize(cascade_number);
	for(int i = 0; i < cascade_number; ++i)
	{
		std::cout << i + 1 << " cascade loaded." << std::endl;
		Regressor regressor(
			tree_number,
			multiple_trees_number,
			tree_depth,
			feature_number_of_node,
			feature_pool_size,
			shrinkage_factor,
			padding,
			lambda);

		for(int j = 0; j < tree_number; ++j)
		{
			auto& tree = regressor.trees()[j].model();
			tree.split_model.resize(root_number);
			tree.residual_model.resize(leaf_number);

			auto& nodes_json = tree_json["regressor"][i][j]["node"];
			for(int k = 0; k < root_number; ++k)
			{
				auto& node = tree.split_model[k];
				auto& node_json = nodes_json[k];
				node.landmark_index1 = node_json["landmark_index"][0].get<int>();
				node.landmark_index2 = node_json["landmark_index"][1].get<int>();
				node.index1_offset(0) = node_json["offset"][0].get<float>();
				node.index1_offset(1) = node_json["offset"][1].get<float>();
				node.index2_offset(0) = node_json["offset"][2].get<float>();
				node.index2_offset(1) = node_json["offset"][3].get<float>();
				node.threshold = node_json["threshold"].get<float>();
			}

			auto& leaf_json = tree_json["regressor"][i][j]["leaf"];
			for(int k = 0; k < leaf_number; ++k)
			{
				auto& leaf_node_data = tree.residual_model[k];
				leaf_node_data.resize(landmark_number, 2);
				for(int r = 0; r < landmark_number; ++r)
				{
					leaf_node_data(r, 0) = leaf_json[k][2*r  ].get<float>();
					leaf_node_data(r, 1) = leaf_json[k][2*r+1].get<float>();
				}
			}
		}
		
		regressors[i] = regressor;
	}
}

void ERT::load_binary(const std::string& path)
{
	std::ifstream fin(path, std::ios::binary);
	std::cout << "open model." << std::endl;

	int landmark_number = 0;
	fin.read((char*)&cascade_number, sizeof(int));
	fin.read((char*)&tree_number, sizeof(int));
	fin.read((char*)&multiple_trees_number, sizeof(int));
	fin.read((char*)&tree_depth, sizeof(int));
	fin.read((char*)&landmark_number, sizeof(int));
	fin.read((char*)&feature_number_of_node, sizeof(int));
	fin.read((char*)&feature_pool_size, sizeof(int));
	fin.read((char*)&shrinkage_factor, sizeof(float));
	fin.read((char*)&padding, sizeof(float));
	fin.read((char*)&lambda, sizeof(float));

	int root_number = get_root_number();
	int leaf_number = get_leaf_number();

	global_mean_landmarks.resize(landmark_number, 2);
	fin.read((char*)global_mean_landmarks.data(), sizeof(float) * landmark_number * 2);

	regressors.resize(cascade_number);
	for(int i = 0; i < cascade_number; ++i)
	{
		std::cout << i + 1 << " cascade loaded." << std::endl;
		Regressor regressor(
			tree_number,
			multiple_trees_number,
			tree_depth,
			feature_number_of_node,
			feature_pool_size,
			shrinkage_factor,
			padding,
			lambda);

		for(int j = 0; j < tree_number; ++j)
		{
			auto& tree = regressor.trees()[j].model();

			tree.split_model.resize(root_number);
			tree.residual_model.resize(leaf_number);

			for(int k = 0; k < root_number; ++k)
			{
				auto& node = tree.split_model[k];
				fin.read((char*)&node, sizeof(Node));
			}

			for(int k = 0; k < leaf_number; ++k)
			{
				auto& leaf = tree.residual_model[k];
				leaf.resize(landmark_number, 2);
				fin.read((char*)leaf.data(), sizeof(float) * landmark_number * 2);
			}
		}
		
		regressors[i] = regressor;
	}
	fin.close();
}

void ERT::find_landmark(const cv::Mat_<uchar> image, const Eigen::Vector4f& face_rect, Eigen::MatrixX2f& landmark) const
{
	Eigen::MatrixX2f bbox(4, 2);
  	Eigen::MatrixX2f bbox_normal(4, 2);

	bbox(0, 0) = face_rect(0);		bbox(0, 1) = face_rect(2);
	bbox(1, 0) = face_rect(1);		bbox(1, 1) = face_rect(2);
	bbox(2, 0) = face_rect(0);		bbox(2, 1) = face_rect(3);
	bbox(3, 0) = face_rect(1);		bbox(3, 1) = face_rect(3);
	
  	bbox_normal(0, 0) = 0; 	bbox_normal(0, 1) = 0;
	bbox_normal(1, 0) = 1; 	bbox_normal(1, 1) = 0;
	bbox_normal(2, 0) = 0; 	bbox_normal(2, 1) = 1;
	bbox_normal(3, 0) = 1; 	bbox_normal(3, 1) = 1; 

	Eigen::Matrix2f scale_rotate_normal_to_image;
	Eigen::RowVector2f translation_normal_to_image;
	Eigen::Matrix2f scale_rotate_mean_to_cur;

	Eigen::MatrixX2f landmarks_cur_normalization = global_mean_landmarks;
	compute_similarity_transform(bbox, bbox_normal, scale_rotate_normal_to_image, translation_normal_to_image);

	int root_number = get_root_number();
	int leaf_number = get_leaf_number();
	int landmark_number = get_landmark_number();

	for(int i = 0; i < cascade_number; ++i)
	{
		Eigen::RowVector2f translation;
		compute_similarity_transform(landmarks_cur_normalization, global_mean_landmarks, scale_rotate_mean_to_cur, translation);
		int times = tree_number / multiple_trees_number;
		for(int j = 0; j < times; ++j)
		{
			Eigen::MatrixX2f residual(landmark_number, 2);
			residual.setZero();
			for(int k = 0; k < multiple_trees_number; ++k)
			{
				int index = 0;
				auto& tree = regressors[i].trees()[j * multiple_trees_number + k].model();
				for(int h = 0; h < root_number; h = index)
				{
					bool left_node = tree.split_model[h].evaluate(
						image,
						landmarks_cur_normalization,
						scale_rotate_mean_to_cur,
						scale_rotate_normal_to_image,
						translation_normal_to_image
						);
					index = 2 * h + (left_node ? 1 : 2);
				}
			
				residual += tree.residual_model[index - root_number];
				index = 0;
			}
			landmarks_cur_normalization += shrinkage_factor * (residual / multiple_trees_number);
		}
	}
	if (landmark.rows() != landmarks_cur_normalization.rows())
		landmark.resize(landmarks_cur_normalization.rows(), 2);
	normalization(landmark, landmarks_cur_normalization, scale_rotate_normal_to_image, translation_normal_to_image);
}

void ERT::find_landmark(const cv::Mat_<uchar> image, Eigen::MatrixX2f& landmark) const
{
	Eigen::Matrix2f scale_rotate_normal_to_image;
	Eigen::RowVector2f translation_normal_to_image;
	Eigen::Matrix2f scale_rotate_mean_to_cur;

	Eigen::MatrixX2f landmarks_cur_normalization = global_mean_landmarks;
	compute_similarity_transform(landmark, global_mean_landmarks, scale_rotate_normal_to_image, translation_normal_to_image);

	int root_number = get_root_number();
	int leaf_number = get_leaf_number();
	int landmark_number = get_landmark_number();

	for(int i = 0; i < cascade_number; ++i)
	{
		Eigen::RowVector2f translation;
		compute_similarity_transform(landmarks_cur_normalization, global_mean_landmarks, scale_rotate_mean_to_cur, translation);
		int times = tree_number / multiple_trees_number;
		for(int j = 0; j < times; ++j)
		{
			Eigen::MatrixX2f residual(landmark_number, 2);
			residual.setZero();
			for(int k = 0; k < multiple_trees_number; ++k)
			{
				int index = 0;
				auto& tree = regressors[i].trees()[j * multiple_trees_number + k].model();
				for(int h = 0; h < root_number; h = index)
				{
					bool left_node = tree.split_model[h].evaluate(
						image,
						landmarks_cur_normalization,
						scale_rotate_mean_to_cur,
						scale_rotate_normal_to_image,
						translation_normal_to_image
						);
					index = 2 * h + (left_node ? 1 : 2);
				}
			
				residual += tree.residual_model[index - root_number];
				index = 0;
			}
			landmarks_cur_normalization += shrinkage_factor * (residual / multiple_trees_number);
		}
	}
	if (landmark.rows() != landmarks_cur_normalization.rows())
		landmark.resize(landmarks_cur_normalization.rows(), 2);
	normalization(landmark, landmarks_cur_normalization, scale_rotate_normal_to_image, translation_normal_to_image);
}