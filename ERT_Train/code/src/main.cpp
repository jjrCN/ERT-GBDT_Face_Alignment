#ifdef WIN32
#define NOMINMAX
#endif

#include <ERT.hpp>
#include <ghc/filesystem.hpp>
#include <chrono>

#include <clipp.h>

namespace fs = ghc::filesystem;
using namespace ert;

int main(int argc, char** argv)
{
	bool help = false;

	int cascade_number = 10;
	int tree_number = 500;
	int multiple_trees_number = 1;
	int tree_depth = 5;
	int initialization = 20;
	int feature_number_of_node = 20;
	int feature_pool_size = 400;
	float shrinkage_factor = 0.1f;
	float padding = 0.1f;
	float lambda = 0.1f;

    std::string train_data_directory;
	std::string validation_data_directory;
    std::string output_directory = "./result";
	std::string face_detector_path = "./facedetection/haarcascade_frontalface_alt2.xml";

    auto cli = (
        (clipp::required("-i", "--input") & clipp::value("input training data directory").set(train_data_directory)),
        (clipp::option("-o", "--output") & clipp::value("output directory").set(output_directory)),
        (clipp::option("-v", "--validation") & clipp::value("input validation data directory").set(validation_data_directory)),
        clipp::option("-h", "--help").set(help, true),
		(clipp::option("-f", "--face-detector") & clipp::value("face detector xml path").set(face_detector_path)),
		(clipp::option("-c", "--cascade") & clipp::value("cascade count").set(cascade_number)),
		(clipp::option("-t", "--tree") & clipp::value("tree count").set(tree_number)),
		(clipp::option("-m", "--multiple-tree") & clipp::value("multiple tree count").set(multiple_trees_number)),
		(clipp::option("-n", "--node") & clipp::value("feature number of node").set(feature_number_of_node)),
		(clipp::option("-p", "--pool") & clipp::value("feature pool size").set(feature_pool_size)),
		(clipp::option("-z", "--initialization") & clipp::value("initialization count").set(initialization)),
		(clipp::option("-s", "--shrinkage") & clipp::value("shrinkage factor").set(shrinkage_factor)),
		(clipp::option("-d", "--padding") & clipp::value("padding").set(padding)),
		(clipp::option("-l", "--lambda") & clipp::value("lambda").set(lambda))
    );

	auto parse_result = clipp::parse(argc, argv, cli);
    if (!parse_result || help) {
        std::cout << clipp::make_man_page(cli, argv[0]);
        return -1;
    }

	std::srand((uint32_t)std::chrono::high_resolution_clock::now().time_since_epoch().count());

	//============load image======================//
	std::cout << "[Stage 1] Loading image" << std::endl;

	std::vector<Sample> train_data, validation_data;
	if (validation_data_directory.empty()) {
		std::vector<Sample> all_data;
		load_samples(all_data, train_data_directory, face_detector_path);
		std::set<int> validation_data_indices;
		int validation_data_count = (int)all_data.size() / 10;
		while (validation_data_indices.size() < validation_data_count) {
			validation_data_indices.insert(std::rand() % all_data.size());
		}
		for (int i = 0; i < (int)all_data.size(); i++) {
			if (validation_data_indices.find(i) != validation_data_indices.end()) {
				validation_data.push_back(all_data[i]);
			}
			else {
				train_data.push_back(all_data[i]);
			}
		}
		std::cout << train_data.size() << " images for training" << std::endl;
		std::cout << validation_data.size() << " images for validation" << std::endl;
	}
	else {
		load_samples(train_data, train_data_directory, face_detector_path);
		load_samples(validation_data, validation_data_directory, face_detector_path);
	}

//============data preprocessing==============//
	std::cout << "[Stage 2] Generating train_data" << std::endl;
	generate_train_data(train_data, initialization);

//============create ERT======================//
	std::cout << "[Stage 3] Training ERT" << std::endl;

	ERT FaceAlignmentOperator(cascade_number, tree_number, multiple_trees_number, tree_depth, 
		feature_number_of_node, feature_pool_size, shrinkage_factor, padding, initialization, lambda);

	FaceAlignmentOperator.train(train_data, validation_data, output_directory);

	auto model_path = fs::path(output_directory) / "model";
	fs::remove_all(model_path);
	fs::create_directories(model_path);
	FaceAlignmentOperator.save(model_path / "ERT.json");
	FaceAlignmentOperator.save_binary(model_path / "ERT.bin");

	auto output_train_path = fs::path(output_directory) / "train_result";
	fs::remove_all(output_train_path);
	fs::create_directories(output_train_path);
	for(int i = 0; i < train_data.size(); ++i)
		output(train_data[i], output_train_path);

	auto output_validation_path = fs::path(output_directory) / "validation_result";
	fs::remove_all(output_validation_path);
	fs::create_directories(output_validation_path);
	for(int i = 0; i < validation_data.size(); ++i)
		output(validation_data[i], output_validation_path);	
}