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

using namespace nlohmann;

class UnLeafNode{
public:
	int landmark_index1;
	int landmark_index2;
	Eigen::RowVector2f index1_offset;
	Eigen::RowVector2f index2_offset;
	float threshold;
};

class TreeModel{
public:
	std::vector<UnLeafNode> splite_model;
	std::vector<Eigen::MatrixX2f> residual_model;
};

void compute_scale_rotate_transform(
	const Eigen::MatrixX2f &target,
	const Eigen::MatrixX2f &origin,
	Eigen::Matrix2f &scale_rotate,
	Eigen::RowVector2f &transform)
{
	int rows = (int)origin.rows();
	int cols = (int)origin.cols();

	Eigen::MatrixX3f origin_new(rows, 3);
	origin_new.block(0, 0, rows, 2) = origin;
	origin_new.block(0, 2, rows, 1) = Eigen::VectorXf::Ones(rows);

	auto pinv = (origin_new.transpose() * origin_new).inverse() * origin_new.transpose();
	auto weight = pinv * target;

	scale_rotate(0, 0) = weight(0, 0);
	scale_rotate(0, 1) = weight(0, 1);
	scale_rotate(1, 0) = weight(1, 0);
	scale_rotate(1, 1) = weight(1, 1);

	transform(0, 0) = weight(2, 0);
	transform(0, 1) = weight(2, 1);
}

void compute_scale_rotate(
	const Eigen::MatrixX2f &target,
	const Eigen::MatrixX2f &origin,
	Eigen::Matrix2f &scale_rotate)
{
	int rows = (int)origin.rows();
	int cols = (int)origin.cols();

	Eigen::MatrixX3f origin_new(rows, 3);
	origin_new.block(0, 0, rows, 2) = origin;
	origin_new.block(0, 2, rows, 1) = Eigen::VectorXf::Ones(rows);

	auto pinv = (origin_new.transpose() * origin_new).inverse() * origin_new.transpose();
	auto weight = pinv * target;

	scale_rotate(0, 0) = weight(0, 0);
	scale_rotate(0, 1) = weight(0, 1);
	scale_rotate(1, 0) = weight(1, 0);
	scale_rotate(1, 1) = weight(1, 1);
}

void normalization(cv::Mat_<float> &target, const cv::Mat_<float> &origin, const cv::Mat_<float> &scale_rotate, const cv::Mat_<float> &transform)
{
	cv::Mat_<float> ones = cv::Mat_<float>::ones(origin.rows, 1);
	cv::Mat_<float> temp1;
	cv::hconcat(origin, ones, temp1);
	cv::Mat_<float> temp2;
	cv::vconcat(scale_rotate, transform, temp2);
	target = temp1 * temp2;
}

void normalization2(
	Eigen::MatrixX2f &target,
	const Eigen::MatrixX2f &origin,
	const Eigen::Matrix2f &scale_rotate,
	const Eigen::RowVector2f &transform)
{
	target = (origin * scale_rotate).rowwise() + transform;
}

int main(int argc, char* argv[])
{
	std::srand((uint32_t)std::chrono::high_resolution_clock::now().time_since_epoch().count());

	std::string model_path = "./result/model/ERT.json";
	std::ifstream fin(model_path);
	json tree_json;
	fin >> tree_json;
	fin.close();

	std::cout << "open model." << std::endl;

	int cascade_number = tree_json["info"]["cascade_number"].get<int>();
	int tree_number = tree_json["info"]["tree_number"].get<int>();
	int multiple_trees_number = tree_json["info"]["multiple_trees_number"].get<int>();
	int tree_depth = tree_json["info"]["tree_depth"].get<int>();
	int landmark_number = tree_json["info"]["landmark_number"].get<int>();
	float shrinkage_factor = tree_json["info"]["shrinkage_factor"].get<float>();

	int root_number = (int)(std::pow(2, tree_depth - 1) - 1);
	int leaf_number = (int)(std::pow(2, tree_depth - 1));

	Eigen::MatrixX2f global_mean_landmarks(landmark_number, 2);
	for(int i = 0; i < landmark_number; ++i)
	{
		global_mean_landmarks(i, 0) = tree_json["global_mean_landmark"][2*i  ].get<float>();
		global_mean_landmarks(i, 1) = tree_json["global_mean_landmark"][2*i+1].get<float>();
	}

	std::vector<std::vector<TreeModel> > regressors(cascade_number);
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
	Eigen::RowVector2f u_cur;
	Eigen::RowVector2f v_cur;
	Eigen::RowVector2f u_data;
	Eigen::RowVector2f v_data;
	Eigen::RowVector2f u_data_unnormalization;
	Eigen::RowVector2f v_data_unnormalization;
	int u_index, v_index;
	int u_value, v_value;
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
		haar_cascade.detectMultiScale(image, faces_temp, 1.1, 2, 0, cv::Size(30, 30));
		time_detect_end = clock();
		if(!faces_temp.empty())
		{
			GTBox_Rect = faces_temp[0];
			GTBox(0, 0) = (float)(GTBox_Rect.x);
			GTBox(0, 1) = (float)(GTBox_Rect.y);
			GTBox(1, 0) = (float)(GTBox_Rect.x + GTBox_Rect.width);
			GTBox(1, 1) = (float)(GTBox_Rect.y);
			GTBox(2, 0) = (float)(GTBox_Rect.x);
			GTBox(2, 1) = (float)(GTBox_Rect.y + GTBox_Rect.height);
			GTBox(3, 0) = (float)(GTBox_Rect.x + GTBox_Rect.width);
			GTBox(3, 1) = (float)(GTBox_Rect.y + GTBox_Rect.height);
			Eigen::MatrixX2f landmarks_cur_normalization = global_mean_landmarks;
			time_rotate_trainsform_begin = clock();
			compute_scale_rotate_transform(GTBox, GTBox_normalization, scale_rotate_normalization_to_truth, transform_normalization_to_truth);
			time_rotate_trainsform_end = clock();
			time_begin = clock();
			for(int i = 0; i < cascade_number; ++i)
			{
				time_one_cascade_begin = clock();
				compute_scale_rotate(landmarks_cur_normalization, global_mean_landmarks, scale_rotate_from_mean_to_cur);
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
							u_index = tree.splite_model[h].landmark_index1;
							v_index = tree.splite_model[h].landmark_index2;
							u_cur = landmarks_cur_normalization.row(u_index);
							v_cur = landmarks_cur_normalization.row(v_index);
							u_data = u_cur + tree.splite_model[h].index1_offset * scale_rotate_from_mean_to_cur;
							v_data = v_cur + tree.splite_model[h].index2_offset * scale_rotate_from_mean_to_cur;
							small_nor_begin = clock();

							u_data_unnormalization = u_data * scale_rotate_normalization_to_truth + transform_normalization_to_truth;
							v_data_unnormalization = v_data * scale_rotate_normalization_to_truth + transform_normalization_to_truth;

							small_nor_end = clock();
							if(u_data_unnormalization(0) < 0 || u_data_unnormalization(0) >= image.cols || 
								u_data_unnormalization(1) < 0 || u_data_unnormalization(1) >= image.rows)
								u_value = 0;
							else
								u_value = image.at<uchar>((int)u_data_unnormalization(1), (int)u_data_unnormalization(0));

							if(v_data_unnormalization(0) < 0 || v_data_unnormalization(0) >= image.cols || 
								v_data_unnormalization(1) < 0 || v_data_unnormalization(1) >= image.rows)
								v_value = 0;
							else
								v_value = image.at<uchar>((int)v_data_unnormalization(1), (int)v_data_unnormalization(0));

							if(u_value - v_value > tree.splite_model[h].threshold)
								index = 2 * h + 1;
							else
								index = 2 * h + 2;
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
			normalization2(landmarks_cur, landmarks_cur_normalization, scale_rotate_normalization_to_truth, transform_normalization_to_truth);
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
