#include <cmath>
#include <sstream>
#include <ctime>
#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/foreach.hpp>
#include <string>
#include <set>
#include <exception>

namespace pt = boost::property_tree;

class UnLeafNode{
public:
	int landmark_index1;
	int landmark_index2;
	cv::Mat_<float> index1_offset;
	cv::Mat_<float> index2_offset;
	float threshold;
};

class TreeModel{
public:
	std::vector<UnLeafNode> splite_model;
	std::vector<cv::Mat_<float>> residual_model;
};

void compute_scale_rotate_transform(const cv::Mat_<float> &target, const cv::Mat_<float> &origin, cv::Mat_<float> &scale_rotate, cv::Mat_<float> &transform)
{
	int rows = origin.rows;
	int cols = origin.cols;
	cv::Mat_<float> ones = cv::Mat_<float>::ones(rows, 1);

	cv::Mat_<float> origin_new;
	cv::hconcat(origin, ones, origin_new);

	cv::Mat_<float> pinv;
	cv::invert(origin_new, pinv, cv::DECOMP_SVD);

	cv::Mat_<float> weight = pinv * target;

	scale_rotate(0, 0) = weight(0, 0);
	scale_rotate(0, 1) = weight(0, 1);
	scale_rotate(1, 0) = weight(1, 0);
	scale_rotate(1, 1) = weight(1, 1);

	transform(0, 0) = weight(2, 0);
	transform(0, 1) = weight(2, 1);
}

void compute_scale_rotate(const cv::Mat_<float> &target, const cv::Mat_<float> &origin, cv::Mat_<float> &scale_rotate)
{
	int rows = origin.rows;
	int cols = origin.cols;
	cv::Mat_<float> ones = cv::Mat_<float>::ones(rows, 1);

	cv::Mat_<float> origin_new;
	cv::hconcat(origin, ones, origin_new);

	cv::Mat_<float> pinv;
	cv::invert(origin_new, pinv, cv::DECOMP_SVD);

	cv::Mat_<float> weight = pinv * target;

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

void normalization2(cv::Mat_<float> &target, const cv::Mat_<float> &origin, const cv::Mat_<float> &scale_rotate, const cv::Mat_<float> &transform)
{
	int rows = origin.rows;
	cv::Mat_<float> temp(1, 2);
	for(int i = 0; i < rows; ++i)
	{
		temp(0, 0) = origin(i, 0);
		temp(0, 1) = origin(i, 1);
		temp = temp * scale_rotate + transform;
		target(i, 0) = temp(0, 0);
		target(i, 1) = temp(0, 1);
	}
}

int main(int argc, char* argv[])
{
	std::srand(std::time(0));

	std::string model_path = "./../model/ERT.xml";
	pt::ptree tree_xml;
	pt::read_xml(model_path, tree_xml);
	std::cout << "open model." << std::endl;

	int cascade_number = tree_xml.get<int>("ERT.cascade_number");
	int tree_number = tree_xml.get<int>("ERT.tree_number");
	int multiple_trees_number = tree_xml.get<int>("ERT.multiple_trees_number");
	int tree_depth = tree_xml.get<int>("ERT.tree_depth");
	int landmark_number = tree_xml.get<int>("ERT.landmark_number");
	float shrinkage_factor = tree_xml.get<float>("ERT.shrinkage_factor");

	int root_number = std::pow(2, tree_depth - 1) - 1;
	int leaf_number = std::pow(2, tree_depth - 1);

	cv::Mat_<float> global_mean_landmarks(landmark_number, 2);
	for(int i = 0; i < landmark_number; ++i)
	{
		std::stringstream stream_global_mean_landmarks;
		stream_global_mean_landmarks << i;
		std::string global_mean_landmarks_path_x = "ERT.parameters.global_mean_landmarks.x_" + stream_global_mean_landmarks.str();
		std::string global_mean_landmarks_path_y = "ERT.parameters.global_mean_landmarks.y_" + stream_global_mean_landmarks.str();

		global_mean_landmarks(i, 0) = tree_xml.get<float>(global_mean_landmarks_path_x);
		global_mean_landmarks(i, 1) = tree_xml.get<float>(global_mean_landmarks_path_y);
	}

	std::vector<std::vector<TreeModel> > regressors(cascade_number);
	for(int i = 0; i < cascade_number; ++i)
	{	std::cout << i + 1 << " cascade loaded." << std::endl;
		std::stringstream stream_regressor;
		stream_regressor << i;
		std::string regressor_path = "ERT.parameters.regressor_" + stream_regressor.str() + ".";

		std::vector<TreeModel> regressor(tree_number);
		for(int j = 0; j < tree_number; ++j)
		{
			std::stringstream stream_tree;
			stream_tree << j;
			std::string tree_path = regressor_path + "tree_" + stream_tree.str() + ".";

			TreeModel tree;
			tree.splite_model.resize(root_number);
			tree.residual_model.resize(leaf_number);

			for(int k = 0; k < root_number; ++k)
			{
				std::stringstream stream_root_node;
				stream_root_node << k;
				std::string root_node = tree_path + "root_node_" + stream_root_node.str() + ".";

				std::string landmark_index1_path = root_node + "landmark_index1";
				std::string landmark_index2_path = root_node + "landmark_index2";
				std::string index1_offset_x = root_node + "index1_offset_x";
				std::string index1_offset_y = root_node + "index1_offset_y";
				std::string index2_offset_x = root_node + "index2_offset_x";
				std::string index2_offset_y = root_node + "index2_offset_y";
				std::string threshold = root_node + "threshold";
 
				tree.splite_model[k].landmark_index1 = tree_xml.get<int>(landmark_index1_path);
				tree.splite_model[k].landmark_index2 = tree_xml.get<int>(landmark_index2_path);
				tree.splite_model[k].index1_offset = cv::Mat_<float>::zeros(1, 2);
				tree.splite_model[k].index2_offset = cv::Mat_<float>::zeros(1, 2);
				tree.splite_model[k].index1_offset(0, 0) = tree_xml.get<float>(index1_offset_x);
				tree.splite_model[k].index1_offset(0, 1) = tree_xml.get<float>(index1_offset_y);
				tree.splite_model[k].index2_offset(0, 0) = tree_xml.get<float>(index2_offset_x);
				tree.splite_model[k].index2_offset(0, 1) = tree_xml.get<float>(index2_offset_y);
				tree.splite_model[k].threshold = tree_xml.get<float>(threshold);
			}

			for(int k = 0; k < leaf_number; ++k)
			{
				std::stringstream stream_root_node;
				stream_root_node << k;
				std::string root_node = tree_path + "leaf_node_" + stream_root_node.str() + ".";

				cv::Mat_<float> leaf_node_data = cv::Mat_<float>::zeros(landmark_number, 2);

				for(int r = 0; r < landmark_number; ++r)
				{
					std::stringstream landmark_index;
					landmark_index << r;
					std::string residual_x = root_node + "x_" + landmark_index.str();
					std::string residual_y = root_node + "y_" + landmark_index.str();

					leaf_node_data(r, 0) = tree_xml.get<float>(residual_x);
					leaf_node_data(r, 1) = tree_xml.get<float>(residual_y);
				}
				tree.residual_model[k] = leaf_node_data.clone();
			}
			regressor[j] = tree;
		}
		
		regressors[i] = regressor;
	}

  	cv::Mat_<uchar> image;
  	cv::Rect GTBox_Rect;
  	cv::Mat_<float> GTBox(4, 2);
  	cv::Mat_<float> GTBox_normalization(4, 2);
  	GTBox_normalization(0, 0) = 0; GTBox_normalization(0, 1) = 0;
	GTBox_normalization(1, 0) = 1; GTBox_normalization(1, 1) = 0;
	GTBox_normalization(2, 0) = 0; GTBox_normalization(2, 1) = 1;
	GTBox_normalization(3, 0) = 1; GTBox_normalization(3, 1) = 1; 

	cv::Mat_<float> scale_rotate_normalization_to_truth(2, 2);
	cv::Mat_<float> transform_normalization_to_truth(1, 2);

	cv::Mat_<float> scale_rotate_from_mean_to_cur(2, 2);
	cv::Mat_<float> transform_from_mean_to_cur(1, 2);

	cv::Mat_<float> landmarks_cur_normalization = global_mean_landmarks.clone();
	cv::Mat_<float> landmarks_cur(landmark_number, 2);
	cv::Mat_<float> u_cur(1, 2);
	cv::Mat_<float> v_cur(1, 2);
	cv::Mat_<float> u_data(1, 2);
	cv::Mat_<float> v_data(1, 2);
	cv::Mat_<float> u_data_unnormalization(1, 2);
	cv::Mat_<float> v_data_unnormalization(1, 2);
	int u_index, v_index;
	int u_value, v_value;
	int index = 0;

	std::string haar_feature = "../haarcascade_frontalface_alt2.xml";
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
    	std::cout << "Vedio open failed. please check your vedio equitment." << std::endl;
    	exit(0);
  	}
  	std::cout << "open vedio." << std::endl;
  	int m = 1;
  	while(m)
  	{
  		time_total_begin = clock();
  		cap >> image;

  		std::vector<cv::Rect> faces_temp;
  		time_detect_begin = clock();
		haar_cascade.detectMultiScale(image, faces_temp, 1.1, 2, 0, cv::Size(30, 30));
		time_detect_end = clock();
		if(!faces_temp.empty())
		{
			GTBox_Rect = faces_temp[0];
			GTBox(0, 0) = GTBox_Rect.x; GTBox(0, 1) = GTBox_Rect.y;
			GTBox(1, 0) = GTBox_Rect.x + GTBox_Rect.width; GTBox(1, 1) = GTBox_Rect.y;
			GTBox(2, 0) = GTBox_Rect.x; GTBox(2, 1) = GTBox_Rect.y + GTBox_Rect.height;
			GTBox(3, 0) = GTBox_Rect.x + GTBox_Rect.width; GTBox(3, 1) = GTBox_Rect.y + GTBox_Rect.height;
			cv::Mat_<float> landmarks_cur_normalization = global_mean_landmarks.clone();
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
					cv::Mat_<float> residual = cv::Mat_<float>::zeros(landmark_number, 2);
					for(int k = 0; k < multiple_trees_number; ++k)
					{
						time_one_tree_begin = clock();
						TreeModel &tree = regressors[i][j * multiple_trees_number + k];
						for(int h = 0; h < root_number; h = index)
						{
							splite_begin = clock();
							u_index = tree.splite_model[h].landmark_index1;
							v_index = tree.splite_model[h].landmark_index2;
							u_cur(0, 0) = landmarks_cur_normalization(u_index, 0); u_cur(0, 1) = landmarks_cur_normalization(u_index, 1);
							v_cur(0, 0) = landmarks_cur_normalization(v_index, 0); v_cur(0, 1) = landmarks_cur_normalization(v_index, 1);
							u_data = u_cur + tree.splite_model[h].index1_offset * scale_rotate_from_mean_to_cur;
							v_data = v_cur + tree.splite_model[h].index2_offset * scale_rotate_from_mean_to_cur;
							small_nor_begin = clock();

							u_data_unnormalization = u_data * scale_rotate_normalization_to_truth + transform_normalization_to_truth;
							v_data_unnormalization = v_data * scale_rotate_normalization_to_truth + transform_normalization_to_truth;

							small_nor_end = clock();
							if(u_data_unnormalization(0, 0) < 0 || u_data_unnormalization(0, 0) > image.cols || 
								u_data_unnormalization(0, 1) < 0 || u_data_unnormalization(0, 1) > image.rows)
								u_value = 0;
							else
								u_value = image.at<uchar>((int)u_data_unnormalization(0, 1), (int)u_data_unnormalization(0, 0));

							if(v_data_unnormalization(0, 0) < 0 || v_data_unnormalization(0, 0) > image.cols || 
								v_data_unnormalization(0, 1) < 0 || v_data_unnormalization(0, 1) > image.rows)
								v_value = 0;
							else
								v_value = image.at<uchar>((int)v_data_unnormalization(0, 1), (int)v_data_unnormalization(0, 0));

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
				float x = landmarks_cur(i, 0);
				float y = landmarks_cur(i, 1);
				cv::circle(image, cv::Point(x, y), 1, cv::Scalar(255, 255, 255), -1);
			}
			cv::rectangle(image, GTBox_Rect, cv::Scalar(255, 255, 255), 1, 1, 0);
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
		cv::imshow("face", image);
	   	cv::waitKey(40);
  	}
}
