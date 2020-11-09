#include <ERT.hpp>
#include <ghc/filesystem.hpp>
#include <chrono>

namespace fs = ghc::filesystem;
using namespace ert;

int main(int argc, char* argv[])
{
	std::srand((uint32_t)std::chrono::high_resolution_clock::now().time_since_epoch().count());
//============load image======================//
	std::cout << "[Stage 1] Loading image" << std::endl;
	std::vector<Sample> traindata;
	std::string trainpath = "./dataset/lfpw/trainset/";
	load_samples(traindata, trainpath);

	std::vector<Sample> validationdata;
	std::string validationpath = "./dataset/lfpw/testset/";
	load_samples(validationdata, validationpath);

//============data preprocessing==============//
	std::cout << "[Stage 2] Generating traindata" << std::endl;
	int initialization = 20;
	generate_train_data(traindata, initialization);

//============create ERT======================//
	std::cout << "[Stage 3] Training ERT" << std::endl;
	int cascade_number = 10;
	int tree_number = 500;
	int multiple_trees_number = 1;
	int tree_depth = 5;
	int feature_number_of_node = 20;
	int feature_pool_size = 400;
	float shrinkage_factor = 0.1f;
	float padding = 0.1f;
	float lambda = 0.1f;

	ERT FaceAlignmentOperator(cascade_number, tree_number, multiple_trees_number, tree_depth, 
		feature_number_of_node, feature_pool_size, shrinkage_factor, padding, initialization, lamda);

	FaceAlignmentOperator.train(traindata, validationdata);

	std::string model_file_path = "./result/model/ERT.json";
	std::string model_file_path_bin = "./result/model/ERT.bin";
	fs::remove_all("./result/model");
	fs::create_directories("./result/model");
	FaceAlignmentOperator.save(model_file_path);
	FaceAlignmentOperator.save_binary(model_file_path_bin);

	std::string outputpath_train = "./result/train_result";

	fs::remove_all(outputpath_train);
	fs::create_directories(outputpath_train);
	for(int i = 0; i < traindata.size(); ++i)
		output(traindata[i], outputpath_train);

	std::string outputpath_vali = "./result/validation_result";
	fs::remove_all(outputpath_vali);
	fs::create_directories(outputpath_vali);
	for(int i = 0; i < validationdata.size(); ++i)
		output(validationdata[i], outputpath_vali);	
}