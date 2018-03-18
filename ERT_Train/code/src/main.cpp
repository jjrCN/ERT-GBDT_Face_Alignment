#include <ERT.hpp>

int main(int argc, char* argv[])
{
	std::srand(std::time(0));
//============load image======================//
	std::cout << "[Stage 1] Loading image" << std::endl;
	std::vector<sample> traindata;
	std::string trainpath = "./../../../dataset/lfpw/trainset/";
	Loadimages(traindata, trainpath);

	std::vector<sample> validationdata;
	std::string validationpath = "./../../../dataset/lfpw/testset/";
	Loadimages(validationdata, validationpath);

//============data preprocessing==============//
	std::cout << "[Stage 2] Generating traindata" << std::endl;
	int initialization = 20;
	GenerateTraindata(traindata, initialization);

//============create ERT======================//
	std::cout << "[Stage 3] Training ERT" << std::endl;
	int cascade_number = 10;
	int tree_number = 500;
	int multiple_trees_number = 1;
	int tree_depth = 5;
	int feature_number_of_node = 20;
	int feature_pool_size = 400;
	float shrinkage_factor = 0.1;
	float padding = 0.1;
	float lamda = 0.1;

	ERT FaceAlignmentOperator(cascade_number, tree_number, multiple_trees_number, tree_depth, 
		feature_number_of_node, feature_pool_size, shrinkage_factor, padding, initialization, lamda);

	FaceAlignmentOperator.train(traindata, validationdata);

	std::string model_file_path = "./../model/ERT_jjr.xml";
	FaceAlignmentOperator.save(model_file_path);

	std::string outputpath_train = "./../train_result";
	rmdir(outputpath_train.c_str());
	mkdir(outputpath_train.c_str(), S_IRWXU);
	for(int i = 0; i < traindata.size(); ++i)
		output(traindata[i], outputpath_train);

	std::string outputpath_vali = "./../validation_result";
	rmdir(outputpath_vali.c_str());
	mkdir(outputpath_vali.c_str(), S_IRWXU);
	for(int i = 0; i < validationdata.size(); ++i)
		output(validationdata[i], outputpath_vali);	
}