#include <ERT.hpp>
#include <ghc/filesystem.hpp>
#include <nlohmann/json.hpp>

namespace fs = ghc::filesystem;
using namespace nlohmann;
using namespace ert;

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

	Regressor regressor_template(tree_number, multiple_trees_number, tree_depth, feature_number_of_node, feature_pool_size, shrinkage_factor, padding, lamda);

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
		global_mean_landmarks += data[i].landmarks_truth_normalizaiotn;
	}
	// global_mean_landmarks /= (data.size() / initialization);
	global_mean_landmarks /= ((float)data.size() / (float)initialization);
	std::cout << "Compute global mean landmarks finished." << std::endl << std::endl;

}

void ERT::train(std::vector<Sample> &data, std::vector<Sample> &validationdata)
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
			std::string path = "./result/train_origin_landmark";
			fs::remove_all(path);
			fs::create_directories(path);
			for(int j = 0; j < 10; ++j)
				output(data[j], path);

			path = "./result/validation_origin_landmark";
			fs::remove_all(path);
			fs::create_directories(path);
			for(int j = 0; j < 10; ++j)
				output(validationdata[j], path);			
		}
		std::cout << "[Cascade " << i + 1 << "] Training..." << std::endl;
		regressors[i].train(data, validationdata ,global_mean_landmarks);

		std::stringstream stream;
		stream << i + 1;
		std::string outputpath_train = "./result/train_cascade_" + stream.str();
		std::string outputpath_vali = "./result/validation_cascade_" + stream.str();

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
	int root_number = (int)std::pow(2, tree_depth - 1) - 1;
	int leaf_number = (int)std::pow(2, tree_depth - 1);
	int landmark_number = (int)global_mean_landmarks.rows();

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
	info["lamda"] = lamda;

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
				node_json[k]["landmark_index"][0] = regressors[i].trees()[j].model()->splite_model[k].landmark_index1;
				node_json[k]["landmark_index"][1] = regressors[i].trees()[j].model()->splite_model[k].landmark_index2;
				node_json[k]["offset"][0] = regressors[i].trees()[j].model()->splite_model[k].index1_offset(0);
				node_json[k]["offset"][1] = regressors[i].trees()[j].model()->splite_model[k].index1_offset(1);
				node_json[k]["offset"][2] = regressors[i].trees()[j].model()->splite_model[k].index2_offset(0);
				node_json[k]["offset"][3] = regressors[i].trees()[j].model()->splite_model[k].index2_offset(1);
				node_json[k]["threshold"] = regressors[i].trees()[j].model()->splite_model[k].threshold;
			}

			auto& leaves_json = forest_json[j]["leaf"];
			for(int k = 0; k < leaf_number; ++k)
			{
				auto& leaf_json = leaves_json[k];
				for(int r = 0; r < landmark_number; ++r)
				{
					leaf_json[2*r  ] = regressors[i].trees()[j].model()->residual_model[k](r, 0);
					leaf_json[2*r+1] = regressors[i].trees()[j].model()->residual_model[k](r, 1);
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
	int root_number = (int)std::pow(2, tree_depth - 1) - 1;
	int leaf_number = (int)std::pow(2, tree_depth - 1);
	int landmark_number = (int)global_mean_landmarks.rows();

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
	fout.write((const char*)&lamda, sizeof(float));

	// global landmark mean
	fout.write((const char*)global_mean_landmarks.data(), sizeof(float) * landmark_number * 2);

	// regressors
	for(int i = 0; i < cascade_number; ++i)
	{
		for(int j = 0; j < tree_number; ++j)
		{
			auto model = regressors[i].trees()[j].model();
			for(int k = 0; k < root_number; ++k)
			{
				auto& node = model->splite_model[k];
				fout.write((const char*)&node, sizeof(Node));
			}

			for(int k = 0; k < leaf_number; ++k)
			{
				auto& leaf = model->residual_model[k];
				fout.write((const char*)leaf.data(), sizeof(float) * landmark_number * 2);
			}
		}
	}

	fout.close();
}
