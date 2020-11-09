#include <cmath>
#include <sstream>
#include <iostream>
#include <fstream>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <Eigen/Dense>

#include <string>
#include <set>
#include <exception>

#include <nlohmann/json.hpp>
#include <chrono>

#include <ERT.hpp>

using namespace nlohmann;
using namespace ert;

class Model {
public:
	int cascade_number;
	int tree_number;
	int multiple_trees_number;
	int tree_depth;
	int landmark_number;
	int feature_number_of_node;
	int feature_pool_size;
	float shrinkage_factor;
	float padding;
	float lamda;

	int root_number;
	int leaf_number;

	Eigen::MatrixX2f global_mean_landmarks;
	std::vector<std::vector<TreeModel>> regressors;
};

void load_model_json(const std::string& filepath, Model& model)
{
	std::ifstream fin(filepath);
	json tree_json;
	fin >> tree_json;
	fin.close();

	std::cout << "open model." << std::endl;

	model.cascade_number = tree_json["info"]["cascade_number"].get<int>();
	model.tree_number = tree_json["info"]["tree_number"].get<int>();
	model.multiple_trees_number = tree_json["info"]["multiple_trees_number"].get<int>();
	model.tree_depth = tree_json["info"]["tree_depth"].get<int>();
	model.landmark_number = tree_json["info"]["landmark_number"].get<int>();
	model.shrinkage_factor = tree_json["info"]["shrinkage_factor"].get<float>();

	model.root_number = (int)(std::pow(2, model.tree_depth - 1) - 1);
	model.leaf_number = (int)(std::pow(2, model.tree_depth - 1));

	auto tree_depth = model.tree_depth;
	auto cascade_number = model.cascade_number;
	auto landmark_number = model.landmark_number;
	auto tree_number = model.tree_number;
	auto root_number = model.root_number;
	auto leaf_number = model.leaf_number;

	auto& global_mean_landmarks = model.global_mean_landmarks;
	global_mean_landmarks.resize(landmark_number, 2);
	for(int i = 0; i < landmark_number; ++i)
	{
		global_mean_landmarks(i, 0) = tree_json["global_mean_landmark"][2*i  ].get<float>();
		global_mean_landmarks(i, 1) = tree_json["global_mean_landmark"][2*i+1].get<float>();
	}

	auto& regressors = model.regressors;
	regressors.resize(cascade_number);
	for(int i = 0; i < cascade_number; ++i)
	{
		std::cout << i + 1 << " cascade loaded." << std::endl;
		std::vector<TreeModel> regressor(tree_number);
		for(int j = 0; j < tree_number; ++j)
		{
			TreeModel tree;
			tree.splite_model.resize(root_number);
			tree.residual_model.resize(leaf_number);

			auto& nodes_json = tree_json["regressor"][i][j]["node"];
			for(int k = 0; k < root_number; ++k)
			{
				auto& node = tree.splite_model[k];
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
			regressor[j] = tree;
		}
		
		regressors[i] = regressor;
	}
}

void load_model_binary(const std::string& filepath, Model& model)
{
	std::ifstream fin(filepath, std::ios::binary);
	std::cout << "open model." << std::endl;

	fin.read((char*)&model.cascade_number, sizeof(int));
	fin.read((char*)&model.tree_number, sizeof(int));
	fin.read((char*)&model.multiple_trees_number, sizeof(int));
	fin.read((char*)&model.tree_depth, sizeof(int));
	fin.read((char*)&model.landmark_number, sizeof(int));
	fin.read((char*)&model.feature_number_of_node, sizeof(int));
	fin.read((char*)&model.feature_pool_size, sizeof(int));
	fin.read((char*)&model.shrinkage_factor, sizeof(float));
	fin.read((char*)&model.padding, sizeof(float));
	fin.read((char*)&model.lamda, sizeof(float));

	model.root_number = (int)(std::pow(2, model.tree_depth - 1) - 1);
	model.leaf_number = (int)(std::pow(2, model.tree_depth - 1));

	auto tree_depth = model.tree_depth;
	auto cascade_number = model.cascade_number;
	auto landmark_number = model.landmark_number;
	auto tree_number = model.tree_number;
	auto root_number = model.root_number;
	auto leaf_number = model.leaf_number;

	auto& global_mean_landmarks = model.global_mean_landmarks;
	global_mean_landmarks.resize(landmark_number, 2);
	fin.read((char*)global_mean_landmarks.data(), sizeof(float) * landmark_number * 2);

	auto& regressors = model.regressors;
	regressors.resize(cascade_number);
	for(int i = 0; i < cascade_number; ++i)
	{
		std::cout << i + 1 << " cascade loaded." << std::endl;
		std::vector<TreeModel> regressor(tree_number);
		for(int j = 0; j < tree_number; ++j)
		{
			TreeModel tree;
			tree.splite_model.resize(root_number);
			tree.residual_model.resize(leaf_number);

			for(int k = 0; k < root_number; ++k)
			{
				auto& node = tree.splite_model[k];
				fin.read((char*)&node, sizeof(UnLeafNode));
			}

			for(int k = 0; k < leaf_number; ++k)
			{
				auto& leaf = tree.residual_model[k];
				leaf.resize(landmark_number, 2);
				fin.read((char*)leaf.data(), sizeof(float) * landmark_number * 2);
			}
			regressor[j] = tree;
		}
		
		regressors[i] = regressor;
	}
	fin.close();
}

int main(int argc, char* argv[])
{
	std::srand((uint32_t)std::chrono::high_resolution_clock::now().time_since_epoch().count());

	Model model;
	// load_model_json("./result/model/ERT.json", model);
	load_model_binary("./result/model/ERT.bin", model);

	auto landmark_number = model.landmark_number;
	auto cascade_number = model.cascade_number;
	auto tree_number = model.tree_number;
	auto multiple_trees_number = model.multiple_trees_number;
	auto root_number = model.root_number;
	auto leaf_number = model.leaf_number;
	auto shrinkage_factor = model.shrinkage_factor;

	auto& global_mean_landmarks = model.global_mean_landmarks;
	auto& regressors = model.regressors;

  	cv::Mat_<uchar> image;
  	cv::Rect GTBox_Rect;
  	Eigen::MatrixX2f GTBox(4, 2);
  	Eigen::MatrixX2f GTBox_normalization(4, 2);
  	GTBox_normalization(0, 0) = 0; GTBox_normalization(0, 1) = 0;
	GTBox_normalization(1, 0) = 1; GTBox_normalization(1, 1) = 0;
	GTBox_normalization(2, 0) = 0; GTBox_normalization(2, 1) = 1;
	GTBox_normalization(3, 0) = 1; GTBox_normalization(3, 1) = 1; 

	Eigen::Matrix2f scale_rotate_normalization_to_truth;
	Eigen::RowVector2f transform_normalization_to_truth;

	Eigen::Matrix2f scale_rotate_from_mean_to_cur;
	Eigen::RowVector2f transform_from_mean_to_cur;

	Eigen::MatrixX2f landmarks_cur_normalization = global_mean_landmarks;
	Eigen::MatrixX2f landmarks_cur(landmark_number, 2);
	landmarks_cur.setZero();
	int index = 0;

	std::string haar_feature = "./facedetection/haarcascade_frontalface_alt2.xml";
	cv::CascadeClassifier haar_cascade;
	haar_cascade.load(haar_feature);
	std::cout << "load face detector completed." << std::endl;
	clock_t time_begin;
	clock_t time_end;
	clock_t time_total_begin;
	clock_t time_total_end;
	clock_t time_detect_begin;
	clock_t time_detect_end;
	clock_t time_rotate_trainsform_begin;
	clock_t time_rotate_trainsform_end;
	clock_t time_one_cascade_begin;
	clock_t time_one_cascade_end;
	clock_t time_one_tree_begin;
	clock_t time_one_tree_end;
	clock_t small_nor_begin;
	clock_t small_nor_end;
	clock_t normalization_begin;
	clock_t normalization_end;
	clock_t splite_begin;
	clock_t splite_end;

	cv::VideoCapture cap(0);
  	if(!cap.isOpened())
 	{
    	std::cout << "Video open failed. please check your video equitment." << std::endl;
    	exit(0);
  	}
  	std::cout << "open video." << std::endl;
  	int m = 1;
  	while(m)
  	{
  		time_total_begin = clock();
		cv::Mat3b colorImage;
  		cap >> colorImage;
		cv::cvtColor(colorImage, image, cv::COLOR_BGR2GRAY);

  		std::vector<cv::Rect> faces_temp;
  		time_detect_begin = clock();
		Eigen::RowVector2f bbMin = landmarks_cur.colwise().minCoeff();
		Eigen::RowVector2f bbMax = landmarks_cur.colwise().maxCoeff();
		auto bbSize = bbMax - bbMin;
		auto bbCenter = (bbMax + bbMin) * 0.5f;
		auto bbLength = bbSize.minCoeff();
		auto paddedLength = bbLength * 1.2f;
		bool face_found = false;
		// if (bbSize(0) * bbSize(1) < 100) {
			haar_cascade.detectMultiScale(image, faces_temp, 1.1, 2, 0, cv::Size(30, 30));
			if (!faces_temp.empty()) {
				GTBox_Rect = faces_temp[0];
				face_found = true;
			}
		// }
		// else {
		// 	GTBox_Rect.x = (int)(bbCenter(0) - paddedLength * 0.5f);
		// 	GTBox_Rect.y = (int)(bbCenter(1) - paddedLength * 0.5f - paddedLength * 0.12f);
		// 	GTBox_Rect.width = (int)paddedLength;
		// 	GTBox_Rect.height = (int)paddedLength;
		// 	compute_similarity_transform(landmarks_cur, global_mean_landmarks, scale_rotate_normalization_to_truth, transform_normalization_to_truth);
		// 	// landmarks_cur_normalization = global_mean_landmarks;
		// 	// compute_similarity_transform(GTBox, GTBox_normalization, scale_rotate_normalization_to_truth, transform_normalization_to_truth);
		// 	face_found = true;
		// }
		time_detect_end = clock();
		if(face_found)
		{
			GTBox(0, 0) = (float)(GTBox_Rect.x);
			GTBox(0, 1) = (float)(GTBox_Rect.y);
			GTBox(1, 0) = (float)(GTBox_Rect.x + GTBox_Rect.width);
			GTBox(1, 1) = (float)(GTBox_Rect.y);
			GTBox(2, 0) = (float)(GTBox_Rect.x);
			GTBox(2, 1) = (float)(GTBox_Rect.y + GTBox_Rect.height);
			GTBox(3, 0) = (float)(GTBox_Rect.x + GTBox_Rect.width);
			GTBox(3, 1) = (float)(GTBox_Rect.y + GTBox_Rect.height);
			landmarks_cur_normalization = global_mean_landmarks;
			time_rotate_trainsform_begin = clock();
			compute_similarity_transform(GTBox, GTBox_normalization, scale_rotate_normalization_to_truth, transform_normalization_to_truth);
			time_rotate_trainsform_end = clock();
			time_begin = clock();
			for(int i = 0; i < cascade_number; ++i)
			{
				time_one_cascade_begin = clock();
				Eigen::RowVector2f translation;
				compute_similarity_transform(landmarks_cur_normalization, global_mean_landmarks, scale_rotate_from_mean_to_cur, translation);
				int times = tree_number / multiple_trees_number;
				for(int j = 0; j < times; ++j)
				{
					Eigen::MatrixX2f residual(landmark_number, 2);
					residual.setZero();
					for(int k = 0; k < multiple_trees_number; ++k)
					{
						time_one_tree_begin = clock();
						TreeModel &tree = regressors[i][j * multiple_trees_number + k];
						for(int h = 0; h < root_number; h = index)
						{
							splite_begin = clock();
							bool left_node = tree.splite_model[h].evaluate(
								image,
								landmarks_cur_normalization,
								scale_rotate_from_mean_to_cur,
								scale_rotate_normalization_to_truth,
								transform_normalization_to_truth
								);
							index = 2 * h + (left_node ? 1 : 2);
							splite_end = clock();
						}
					
						residual += tree.residual_model[index - root_number];
						index = 0;
						time_one_tree_end = clock();
					}
					landmarks_cur_normalization += shrinkage_factor * (residual / multiple_trees_number);
				}
				time_one_cascade_end = clock();
			}
			normalization_begin = clock();
			normalization(landmarks_cur, landmarks_cur_normalization, scale_rotate_normalization_to_truth, transform_normalization_to_truth);
			normalization_end = clock();
			time_end = clock();

			for(int i = 0; i < landmark_number; ++i)
			{
				int x = (int)landmarks_cur(i, 0);
				int y = (int)landmarks_cur(i, 1);
				cv::circle(colorImage, cv::Point(x, y), 1, cv::Scalar(255, 255, 255), -1);
			}
			cv::rectangle(colorImage, GTBox_Rect, cv::Scalar(255, 255, 255), 1, 1, 0);
			time_total_end = clock();
			std::cout << "time total : " << 1000 * (time_total_end - time_total_begin) / (float)CLOCKS_PER_SEC << " ms" << std::endl;
			std::cout << "time detect: " << 1000 * (time_detect_end - time_detect_begin) / (float)CLOCKS_PER_SEC << " ms" << std::endl;
			std::cout << "time alignment : " << 1000 * (time_end - time_begin) / (float)CLOCKS_PER_SEC << " ms" << std::endl;
			std::cout << "time cascade : " << 1000 * (time_one_cascade_end - time_one_cascade_begin) << (float)CLOCKS_PER_SEC << " ms" << std::endl;
			std::cout << "time one tree : " << 1000 * (time_one_tree_end - time_one_tree_begin) << (float)CLOCKS_PER_SEC << " ms" << std::endl;
			std::cout << "time one node : " << 1000 * (splite_end - splite_begin) << (float)CLOCKS_PER_SEC << " ms" << std::endl;
			std::cout << "time compute rotate : " << 1000 * (time_rotate_trainsform_end - time_rotate_trainsform_begin) << (float)CLOCKS_PER_SEC << " ms" << std::endl;
			std::cout << "time normalization : " << 1000 * (normalization_end - normalization_begin) << (float)CLOCKS_PER_SEC << " ms" << std::endl;
			std::cout << "time two small normalization : " << 1000 * (small_nor_end - small_nor_begin) << (float)CLOCKS_PER_SEC << " ms" << std::endl;
			std::cout << "====================================================" << std::endl;
		}
		cv::imshow("face", colorImage);
	   	cv::waitKey(1);
		// cv::waitKey(0);
  	}
}
