#include <utils.hpp>
#include <ghc/filesystem.hpp>

namespace fs = ghc::filesystem;

namespace ert {

void get_files(std::vector<std::string> &names, const std::string &path, const std::set<std::string>& extensions)
{
	for (auto p : fs::recursive_directory_iterator(path)) {
		if (p.is_directory())
			continue;
		auto pp = p.path();
		if (extensions.find(pp.extension()) == extensions.end())
			continue;
		names.push_back(pp.relative_path());
	}
}

bool is_detected(const cv::Rect &box, const float &x_max, const float &x_min, const float &y_max, const float &y_min)
{
	float width = x_max - x_min;
	float height = y_max - y_min;

	if(box.height > 1.5 * height || box.height < 0.5 * height)
		return false;
	if(box.width > 1.5 * width || box.width < 0.5 * width)
		return false;

	float lambda = 0.4f;
	if(box.x < x_min - lambda * width || box.x > x_min + lambda * width)
		return false;
	if(box.y < y_min - lambda * height || box.y > y_min + lambda * height)
		return false;
	if(box.x + box.width < x_max - lambda * width || box.x + box.width > x_max + lambda * width)
		return false;
	if(box.y + box.height < y_max - lambda * height || box.y + box.height > y_max + lambda * height)
		return false;

	return true;
}

void load_samples(std::vector<Sample> &data, const std::string &path, const std::string& face_detector_path)
{
	std::vector<std::string> images_name;
	get_files(images_name, path, {".png", ".jpg", ".jpeg"});
	int images_number = (int)images_name.size();
	cv::Mat_<uchar> image;

	cv::CascadeClassifier haar_cascade;
	haar_cascade.load(face_detector_path);

	for(int i = 0; i < images_number; ++i)
	{
		fs::path image_path = images_name[i];
		image = cv::imread(image_path.c_str(), cv::IMREAD_GRAYSCALE);
		
		auto label_path = image_path.parent_path() / (std::string(image_path.stem()) + ".pts");
		Eigen::MatrixX2f landmark;
		load_pts(label_path, landmark);

		Eigen::RowVector2f bbMin = landmark.colwise().minCoeff();
		Eigen::RowVector2f bbMax = landmark.colwise().maxCoeff();

		std::vector<cv::Rect> faces_temp;
		haar_cascade.detectMultiScale(image, faces_temp, 1.1, 2, 0, cv::Size(30, 30));

		for(int k = 0; k < faces_temp.size(); ++k)
		{
			if (is_detected(faces_temp[k], bbMax(0), bbMin(0), bbMax(1), bbMin(1)))
			{
				Sample temp;
				temp.image_name = images_name[i];
				temp.image = image.clone();
				temp.GTBox = faces_temp[k];
				temp.landmarks_truth = landmark;
				temp.tree_index = 0;
				data.push_back(temp);
			}
		}
		if (i % (images_number / 10) == 0 && i != 0)
			std::cout << 10 * i / (images_number / 10) << "% has finished." << std::endl;
	}

	std::cout << data.size() << " images have been loaded." << std::endl;
	std::cout << "Number of the landmarks : " << data[0].landmarks_truth.rows() << std::endl << std::endl;
}

void load_pts(const std::string& filename, Eigen::MatrixX2f& points)
{
	std::ifstream fin;
	fin.open(filename, std::ios::in);

	std::string temp;
	getline(fin, temp);
	int landmarks_number = 0;
	fin >> temp >> landmarks_number;
	getline(fin, temp);
	getline(fin, temp);

	points.resize(landmarks_number, 2);
	for(int j = 0; j < landmarks_number; ++j)
	{
		float x, y;
		fin >> x >> y;
		points.row(j) = Eigen::Vector2f(x, y);
		getline(fin, temp);
	}
}

void compute_similarity_transform(
	const Eigen::MatrixX2f& target,
	const Eigen::MatrixX2f& origin,
	Eigen::Matrix2f& scale_rotate,
	Eigen::RowVector2f& transform)
{
	Eigen::RowVector2f target_centroid = target.colwise().mean();
	Eigen::RowVector2f origin_centroid = origin.colwise().mean();

	auto centered_target = target.rowwise() - target_centroid;
	auto centered_origin = origin.rowwise() - origin_centroid;

	auto target_scale = sqrtf(centered_target.squaredNorm() / (float)target.rows());
	auto origin_scale = sqrtf(centered_origin.squaredNorm() / (float)origin.rows());

	auto cov_mat = (centered_origin / origin_scale).transpose() * (centered_target / target_scale);
	float angle = atan2f(cov_mat(0, 1) - cov_mat(1, 0), cov_mat(0, 0) + cov_mat(1, 1));

	float scale = target_scale / origin_scale;
	float c = scale * cosf(angle);
	float s = scale * sinf(angle);
	scale_rotate(0, 0) =  c;
	scale_rotate(0, 1) =  s;
	scale_rotate(1, 0) = -s;
	scale_rotate(1, 1) =  c;
	transform = target_centroid - origin_centroid * scale_rotate;
}

void normalization(
	Eigen::MatrixX2f &target,
	const Eigen::MatrixX2f &origin,
	const Eigen::Matrix2f &scale_rotate,
	const Eigen::RowVector2f &transform
	)
{
	target = (origin * scale_rotate).rowwise() + transform;
}

void check_edge(Sample &data)
{
	int rows = data.image.rows;
	int cols = data.image.cols;

	for(int i = 0; i < data.landmarks_truth.rows(); ++i)
	{
		if(data.landmarks_cur(i, 0) < 0)
			data.landmarks_cur(i, 0) = 0.0f;
		if(data.landmarks_cur(i, 0) >= cols)
			data.landmarks_cur(i, 0) = (float)(cols - 1);
		if(data.landmarks_cur(i, 1) < 0)
			data.landmarks_cur(i, 1) = 0.0f;
		if(data.landmarks_cur(i, 1) >= rows)
			data.landmarks_cur(i, 1) = (float)(rows - 1);
	}
}

void generate_validation_data(std::vector<Sample> &data, const Eigen::MatrixX2f &global_mean_landmarks)
{
	Eigen::MatrixX2f target(4, 2);

	target(0, 0) = 0; target(0, 1) = 0;
	target(1, 0) = 1; target(1, 1) = 0;
	target(2, 0) = 0; target(2, 1) = 1;
	target(3, 0) = 1; target(3, 1) = 1;

	Eigen::Matrix2f scale_rotate;
	Eigen::RowVector2f transform;

	Eigen::MatrixX2f origin(4, 2);

	for(int i = 0; i < data.size(); ++i)
	{	
		origin(0, 0) = (float)(data[i].GTBox.x);
		origin(0, 1) = (float)(data[i].GTBox.y);
		origin(1, 0) = (float)(data[i].GTBox.x + data[i].GTBox.width);
		origin(1, 1) = (float)(data[i].GTBox.y);
		origin(2, 0) = (float)(data[i].GTBox.x);
		origin(2, 1) = (float)(data[i].GTBox.y + data[i].GTBox.height);
		origin(3, 0) = (float)(data[i].GTBox.x + data[i].GTBox.width);
		origin(3, 1) = (float)(data[i].GTBox.y + data[i].GTBox.height);

		compute_similarity_transform(target, origin, scale_rotate, transform);

		data[i].scale_rotate_normalization = scale_rotate;
		data[i].transform_normalization = transform;

		data[i].scale_rotate_unnormalization = scale_rotate.inverse();
		data[i].transform_unnormalization = - transform * scale_rotate.inverse();

		normalization(data[i].landmarks_truth_normalization, data[i].landmarks_truth, data[i].scale_rotate_normalization, data[i].transform_normalization);

		data[i].landmarks_cur_normalization = global_mean_landmarks;
		normalization(data[i].landmarks_cur, data[i].landmarks_cur_normalization, 
				data[i].scale_rotate_unnormalization, data[i].transform_unnormalization);
		check_edge(data[i]);
	}
}

void generate_train_data(std::vector<Sample> &data, const int &initialization)
{
	Eigen::MatrixX2f target(4, 2);

	target(0, 0) = 0; target(0, 1) = 0;
	target(1, 0) = 1; target(1, 1) = 0;
	target(2, 0) = 0; target(2, 1) = 1;
	target(3, 0) = 1; target(3, 1) = 1;

	Eigen::Matrix2f scale_rotate;
	Eigen::RowVector2f transform;
	Eigen::MatrixX2f origin(4, 2);

	int data_size_origin = (int)data.size();
	data.resize(initialization * data_size_origin);

	for(int i = 0; i < data_size_origin; ++i)
	{	
		origin(0, 0) = (float)(data[i].GTBox.x);
		origin(0, 1) = (float)(data[i].GTBox.y);
		origin(1, 0) = (float)(data[i].GTBox.x + data[i].GTBox.width);
		origin(1, 1) = (float)(data[i].GTBox.y);
		origin(2, 0) = (float)(data[i].GTBox.x);
		origin(2, 1) = (float)(data[i].GTBox.y + data[i].GTBox.height);
		origin(3, 0) = (float)(data[i].GTBox.x + data[i].GTBox.width);
		origin(3, 1) = (float)(data[i].GTBox.y + data[i].GTBox.height);

		compute_similarity_transform(target, origin, scale_rotate, transform);

		data[i].scale_rotate_normalization = scale_rotate;
		data[i].transform_normalization = transform;

		data[i].scale_rotate_unnormalization = scale_rotate.inverse();
		data[i].transform_unnormalization = - transform * scale_rotate.inverse();

		// compute_similarity_transform(origin, target, scale_rotate, transform);
		// data[i].scale_rotate_unnormalization = scale_rotate;
		// data[i].transform_unnormalization = transform;			

		normalization(data[i].landmarks_truth_normalization, data[i].landmarks_truth, data[i].scale_rotate_normalization, data[i].transform_normalization);
	}

	for(int i = 0; i < data_size_origin; ++i)
	{
		for(int j = 0; j < initialization; ++j)
		{
			auto& train_data = data[i + j * data_size_origin];
			if(j != 0)
				train_data = data[i];

			int index = 0;
			do {
				index = rand() % (data_size_origin);
			} while (index == i);

			train_data.landmarks_cur_normalization = data[index].landmarks_truth_normalization;
			normalization(train_data.landmarks_cur, train_data.landmarks_cur_normalization, 
				train_data.scale_rotate_unnormalization, train_data.transform_unnormalization);
			check_edge(train_data);

			std::stringstream stream;
			stream << i + j * data_size_origin;
			data[i + j * data_size_origin].image_name = stream.str() + "_" + data[i + j * data_size_origin].image_name;
		}
	}

	std::cout << data.size() << " train images have been generated." << std::endl << std::endl;
}

void output(const Sample &data, const std::string &path)
{
	cv::Mat_<uchar> image = data.image.clone();
	cv::rectangle(image, data.GTBox, cv::Scalar(255, 255, 255), 3, 1, 0);

	for(int i = 0; i < data.landmarks_truth.rows(); ++i)
	{
		auto x = (int)data.landmarks_truth(i, 0);
		auto y = (int)data.landmarks_truth(i, 1);
		cv::circle(image, cv::Point(x, y), 1, cv::Scalar(255, 255, 255), -1);

		x = (int)data.landmarks_cur(i, 0);
		y = (int)data.landmarks_cur(i, 1);
		cv::circle(image, cv::Point(x, y), 3, cv::Scalar(0, 0, 0), -1);
	}

	std::string path_image = fs::path(path) / (fs::path(data.image_name).stem().string() + ".jpg");
	cv::imwrite(path_image.c_str(), image);
}

float compute_error(const std::vector<Sample> &data)
{
	const int left_eye_start_idx = 36;
	const int right_eye_start_idx = 42;
	const int eye_landmark_count = 6;

	float total_error = 0;
	for(int i = 0; i < data.size(); ++i)
	{
		auto left_eye = data[i].landmarks_cur.block(left_eye_start_idx, 0, eye_landmark_count, 2).colwise().mean();
		auto right_eye = data[i].landmarks_cur.block(right_eye_start_idx, 0, eye_landmark_count, 2).colwise().mean();
		auto dis = (left_eye - right_eye).norm();
		auto error = (data[i].landmarks_truth - data[i].landmarks_cur).rowwise().norm().mean() / dis;
		total_error += error;
	}

	return total_error / (float)data.size();
}

bool Node::evaluate(
	const cv::Mat_<uchar>& image,
	const Eigen::MatrixX2f& current_normalized_shape,
	const Eigen::Matrix2f& transform_mean_to_normal,
	const Eigen::Matrix2f& transform_normal_to_image,
	const Eigen::RowVector2f& translation_normal_to_image
	) const
{
	Eigen::RowVector2f u_cur = current_normalized_shape.row(landmark_index1);
	Eigen::RowVector2f v_cur = current_normalized_shape.row(landmark_index2);

	Eigen::RowVector2f u_data = u_cur + index1_offset * transform_mean_to_normal;
	Eigen::RowVector2f v_data = v_cur + index2_offset * transform_mean_to_normal;

	Eigen::RowVector2f u_data_image = (u_data * transform_normal_to_image + translation_normal_to_image);
	Eigen::RowVector2f v_data_image = (v_data * transform_normal_to_image + translation_normal_to_image);

	int u_value = 0;
	int v_value = 0;
	if (u_data_image(0) >= 0 && u_data_image(0) < image.cols &&
		u_data_image(1) >= 0 && u_data_image(1) < image.rows)
		u_value = image.at<uchar>((int)u_data_image(1), (int)u_data_image(0));

	if (v_data_image(0) >= 0 && v_data_image(0) < image.cols &&
		v_data_image(1) >= 0 && v_data_image(1) < image.rows)
		v_value = image.at<uchar>((int)v_data_image(1), (int)v_data_image(0));

	return (u_value - v_value) > threshold;
}

} // namespace ert