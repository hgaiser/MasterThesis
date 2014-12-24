//#define DEBUG 1

#include <random>
#include "conversion.h"
#include <boost/python.hpp>

#include "uniform_segment.h"

#include "connection.h"
#include "adjacency.h"
#include "location_prior.h"

#include "random_stopping_criterion.h"

using namespace boost::python;


std::random_device rd_;
std::mt19937 gen_(rd_());

cv::Mat get_bboxes_(const cv::Mat & seg, uint32_t max_size, float sigma, uint32_t n) {
	LocationPrior lp(sigma);

	double max_id_;
	cv::minMaxIdx(seg, nullptr, &max_id_);
	int max_id = max_id_;

	std::vector<std::shared_ptr<Segment>> segments;
	segments.reserve(max_id);
	cv::Size size = seg.size();
	for (int i = 0; i <= max_id; i++) {
		segments.push_back(std::make_shared<UniformSegment>(i, size));
	}

	{
		//AdjacencyMatrix adjacency(max_id + 1);
		for (int i = 0; i < seg.rows; i++) {
			for (int j = 0; j < seg.cols; j++) {
				cv::Point p(j, i);
				uint16_t id = seg.at<uint16_t>(p);
				segments[id]->addPoint(seg, seg, p);

				if (i < seg.rows - 1) {
					uint16_t n = seg.at<uint16_t>(i+1, j);
					if (n != id) {// && adjacency.get(id, n) == false) {
						//adjacency.get(id, n) = true;
						segments[id]->addNeighbour(n);
						segments[n]->addNeighbour(id);
					}
				}

				if (j < seg.cols - 1) {
					uint16_t n = seg.at<uint16_t>(i, j+1);
					if (n != id) { // && adjacency.get(id, n) == false) {
						//adjacency.get(id, n) = true;
						segments[id]->addNeighbour(n);
						segments[n]->addNeighbour(id);
					}
				}
			}
		}
	}

	cv::Mat bboxes;
	float similarity_sum = 0.f;
	for (auto & s: segments) {
		if (s->size == 0)
			continue;
		cv::Mat bbox = cv::Mat(1, 4, CV_32SC1);
		bbox.at<int>(0) = s->min_p.x;
		bbox.at<int>(1) = s->min_p.y;
		bbox.at<int>(2) = s->max_p.x;
		bbox.at<int>(3) = s->max_p.y;
		if (bboxes.empty())
			bboxes = bbox;
		else
			cv::vconcat(bboxes, bbox, bboxes);
	}

	SelectionPriorMap prior;

	prior = lp.computeSelectionPrior(seg, segments);
#ifdef DEBUG
	prior.visualize(seg);
#endif

#ifdef DEBUG
	cv::namedWindow("SelectionLikelihood", cv::WINDOW_AUTOSIZE);
#endif

	for (uint32_t i = 0; i < n; i++) {
		std::shared_ptr<Segment> s = segments[prior.poll()];
// 		std::shared_ptr<Segment> s = segments[rand() % segments.size()];
		uint32_t seg_size = rand() % max_size;
		//RandomStoppingCriterion stop(threshold);
		cv::Rect r(s->min_p, s->max_p);

		for (int j = 0; j < seg_size; j++) {
			if (s->neighbours.size() == 0)
				break;

			auto random_it = std::next(std::begin(s->neighbours), rand() % s->neighbours.size());
			s = s->merge(segments[random_it->first].get());
		}
// 		while (s->neighbours.size() && stop.stop(image, r) == false) {
// #ifdef DEBUG
// 			cv::Mat red = edge * 0.5;
// 			cv::Mat green;
// 			s->mask.copyTo(green);
// 			green *= 255;
// #endif
// 			float sum = 0.f;
// 			std::vector<Connection> connections;
// 			//std::cout << "neighbours:" << std::endl;
// 			for (auto n: s->neighbours) {
// 				connections.push_back(Connection(s->id, n.first, s->computeSimilarity(segments[n.first].get())));
// 				sum += (connections.end() - 1)->similarity;
// 				//std::cout << n << ", ";
// 			}
// 			//std::cout << std::endl;
// 
// #ifdef DEBUG
// 			cv::Mat blue = cv::Mat::zeros(seg.size(), CV_8UC1);
// 			float max_sim = 0.f;
// 			for (auto & c: connections) {
// 				if (c.similarity > max_sim)
// 					max_sim = c.similarity;
// 			}
// 			for (auto & c: connections) {
// 				blue += segments[c.b]->mask * 255 * (c.similarity / max_sim);
// 				std::cout << c.similarity << std::endl;
// 			}
// 
// 			for (int i = 0; i < blue.total(); i++) {
// 				if (blue.at<uint8_t>(i) && green.at<uint8_t>(i)) {
// 				//	std::cout << "blue: " << (int)blue.at<uint8_t>(i) << " green: " << (int)green.at<uint8_t>(i) << std::endl;
// 					for (auto & c: connections) {
// 						if (segments[c.b]->mask.at<uint8_t>(i))
// 							std::cout << "<<<<<<<<<<<<<<<<<<< " << c.b << " <<<<<<<<<<<<<<<<<<<<<" << std::endl;
// 					}
// 				}
// 			}
// 			std::cout << std::endl;
// 			cv::Mat visualization;
// 			cv::merge({blue, green, red}, visualization);
// 			cv::imshow("SelectionLikelihood", visualization);
// 			cv::waitKey();
// #endif
// 
// 			std::uniform_real_distribution<float> dis(0.f, sum);
// 			float rnd = dis(gen_);
// 
// 			sum = 0.f;
// 			for (auto & c: connections) {
// 				sum += c.similarity;
// 				if (sum >= rnd) {
// 					s = s->merge(segments[c.b].get());
// 					break;
// 				}
// 			}
// 		}

		cv::Mat bbox = cv::Mat(1, 4, CV_32SC1);
		bbox.at<int>(0) = s->min_p.x;
		bbox.at<int>(1) = s->min_p.y;
		bbox.at<int>(2) = s->max_p.x;
		bbox.at<int>(3) = s->max_p.y;
		if (bboxes.empty())
			bboxes = bbox;
		else
			cv::vconcat(bboxes, bbox, bboxes);
	}

#ifdef DEBUG
	cv::Mat draw;
	image.copyTo(draw);
	cv::cvtColor(draw, draw, cv::COLOR_BGR2GRAY);
	cv::Rect r(cv::Point(bboxes.at<int>(0), bboxes.at<int>(1)), cv::Point(bboxes.at<int>(2), bboxes.at<int>(3)));
	cv::rectangle(draw, r, cv::Scalar(255));
	cv::addWeighted(draw, 0.5, segments[n - 1]->mask * 255, 0.5, 0.0, draw);
	cv::imshow("Image", draw);

	std::cout << bboxes << std::endl;
#endif

	return bboxes;
}

PyObject * get_bboxes(PyObject * seg_, uint32_t max_size, float sigma, uint32_t n) {
	NDArrayConverter cvt;
	cv::Mat seg     = cvt.toMat(seg_);
	return cvt.toNDArray(get_bboxes_(seg, max_size, sigma, n));
}

static void init_ar() {
	Py_Initialize();
	import_array();
}

BOOST_PYTHON_MODULE(ss_selection) {
	init_ar();

	def("get_bboxes", get_bboxes);
}

// int main(int argc, char * argv[]) {
// 	if (argc != 7) {
// 		std::cout << "Usage: " << argv[0] << " <image> <segmentation> <edge> <iterations> <selection_prior: uniform, location, objectness, objectness-location> <segment_type: uniform, features, objectness>" << std::endl;
// 		return 0;
// 	}
// 
// 	cv::Mat image = cv::imread(argv[1]);
// 	cv::Mat seg = cv::imread(argv[2], cv::IMREAD_UNCHANGED);
// 	cv::Mat edge = cv::imread(argv[3], cv::IMREAD_UNCHANGED);
// 	uint8_t iterations = atoi(argv[4]);
// 	std::string selection_prior = std::string(argv[5]);
// 	std::string segment_type = std::string(argv[6]);
// 
// 	cv::Mat weights(1, 5, CV_32FC1);
// 	weights.at<float>(0) = -0.00697891f;
// 	weights.at<float>(1) = 0.f;
// 	weights.at<float>(2) = 6.65155577f;
// 	weights.at<float>(3) = 1.16613659f;
// 	weights.at<float>(4) = -7.59423175f;
// 	weights = cv::Mat::ones(1, 6, CV_32FC1);
// 
// // 	cv::namedWindow("Image", cv::WINDOW_NORMAL);
// // 	cv::imshow("Image", (seg == 338) * 255);
// // 	cv::waitKey();
// 
// // 	cv::namedWindow("Image", cv::WINDOW_NORMAL);
// //	while (cv::waitKey() != 'q') {
// 		std::clock_t begin = std::clock();
// 		cv::Mat bboxes = get_bboxes_(image, seg, edge, COLOR_SIMILARITY /*| TEXTURE_SIMILARITY*/  | SIZE_SIMILARITY | BBOX_SIMILARITY, weights, iterations, selection_prior, segment_type, 0.85f); 
// 		std::clock_t end = std::clock();
// 		double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
// 		std::cout << "Times passed in seconds: " << elapsed_secs << std::endl;
// //	}
// 
// // 	for (int i = 0; i < bboxes.rows; i++) {
// // 		cv::Point p1(bboxes.at<int>(i,0), bboxes.at<int>(i,1));
// // 		cv::Point p2(bboxes.at<int>(i,2), bboxes.at<int>(i,3));
// // 		cv::Mat tmp;
// // 		image.copyTo(tmp);
// // 		cv::rectangle(tmp, p1, p2, cv::Scalar(255, 255, 0));
// // 		cv::imshow("Image", tmp);
// // 		cv::waitKey();
// // 	}
// 	return 0;
// }
