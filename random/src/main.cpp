//#define DEBUG 1

//#include "CVBoostConverter.hpp"
#include <random>
#include <stdint.h>
#include <functional>
#include "conversion.h"
#include <boost/python.hpp>

#include "uniform_segment.h"
#include "features_segment.h"
#include "objectness_segment.h"

#include "connection.h"
#include "adjacency.h"
#include "uniform_prior.h"
#include "location_prior.h"
#include "objectness_prior.h"
#include "objectness_location_prior.h"

#include "random_stopping_criterion.h"

using namespace boost::python;


std::random_device rd_;
std::mt19937 gen_(rd_());

UniformPrior up;
LocationPrior lp;
ObjectnessPrior op;
ObjectnessLocationPrior olp;

enum SegmentType {
	ST_NONE,
	ST_UNIFORM,
	ST_FEATURES,
	ST_OBJECTNESS,
};

cv::Mat get_bboxes_(const cv::Mat & image, const cv::Mat & seg, const cv::Mat & edge, uint8_t flags, const cv::Mat & weights, uint32_t n, std::string selection_prior, std::string segment_type, float threshold) {
	double max_id_;
	cv::minMaxIdx(seg, nullptr, &max_id_);
	int max_id = max_id_;

	SegmentType st = ST_NONE;
	if (segment_type == "uniform")
		st = ST_UNIFORM;
	else if (segment_type == "features")
		st = ST_FEATURES;
	else if (segment_type == "objectness")
		st = ST_OBJECTNESS;

	if (st == ST_NONE) {
		std::cerr << "Invalid segment type: " << segment_type << std::endl;
		return cv::Mat();
	}

	if (selection_prior == "objectness-slow" || st == ST_OBJECTNESS)
		olp.computeLikelihood(image);

	std::vector<std::shared_ptr<Segment>> segments;
	segments.reserve(max_id);
	int nchannels = image.channels();
	cv::Size size = image.size();
	for (int i = 0; i <= max_id; i++) {
		switch (st) {
		case ST_UNIFORM:
			segments.push_back(std::make_shared<UniformSegment>(i, size));
			break;
		case ST_FEATURES:
			segments.push_back(std::make_shared<FeaturesSegment>(i, nchannels, size, flags, weights));
			break;
		case ST_OBJECTNESS:
			segments.push_back(std::make_shared<ObjectnessSegment>(i, size, olp.likelihood));
			break;
		default: break;
		}
	}

	{
		//AdjacencyMatrix adjacency(max_id + 1);
		for (int i = 0; i < image.rows; i++) {
			for (int j = 0; j < image.cols; j++) {
				cv::Point p(j, i);
				uint16_t id = seg.at<uint16_t>(p);
				segments[id]->addPoint(image, edge, p);

				if (i < image.rows - 1) {
					uint16_t n = seg.at<uint16_t>(i+1, j);
					if (n != id) {// && adjacency.get(id, n) == false) {
						//adjacency.get(id, n) = true;
						segments[id]->addNeighbour(n);
						segments[n]->addNeighbour(id);
					}
				}

				if (j < image.cols - 1) {
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
		s->finalizeSetup();

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

	if (selection_prior == "uniform") {
		prior = up.computeSelectionPrior(image, segments);
	}
	else if (selection_prior == "location") {
		prior = lp.computeSelectionPrior(image, segments);
	}
	else if (selection_prior == "objectness") {
		prior = op.computeSelectionPrior(image, segments);
	}
	else if (selection_prior == "objectness-slow") {
		prior = olp.computeSelectionPrior(image, segments);
	}
	else {
		throw std::runtime_error("Unknown selection prior method: " + selection_prior);
	}
#ifdef DEBUG
	prior.visualize(seg);
#endif

#ifdef DEBUG
	cv::namedWindow("SelectionLikelihood", cv::WINDOW_AUTOSIZE);
#endif

	for (uint32_t i = 0; i < n; i++) {
		std::shared_ptr<Segment> s = segments[prior.poll()];
		RandomStoppingCriterion stop(threshold);
		cv::Rect r(s->min_p, s->max_p);

		while (s->neighbours.size() && stop.stop(image, r) == false) {
#ifdef DEBUG
			cv::Mat red = edge * 0.5;
			cv::Mat green;
			s->mask.copyTo(green);
			green *= 255;
#endif
			float sum = 0.f;
			std::vector<Connection> connections;
			//std::cout << "neighbours:" << std::endl;
			for (auto n: s->neighbours) {
				connections.push_back(Connection(s->id, n.first, s->computeSimilarity(segments[n.first].get())));
				sum += (connections.end() - 1)->similarity;
				//std::cout << n << ", ";
			}
			//std::cout << std::endl;

#ifdef DEBUG
			cv::Mat blue = cv::Mat::zeros(seg.size(), CV_8UC1);
			float max_sim = 0.f;
			for (auto & c: connections) {
				if (c.similarity > max_sim)
					max_sim = c.similarity;
			}
			for (auto & c: connections) {
				blue += segments[c.b]->mask * 255 * (c.similarity / max_sim);
				std::cout << c.similarity << std::endl;
			}

			for (int i = 0; i < blue.total(); i++) {
				if (blue.at<uint8_t>(i) && green.at<uint8_t>(i)) {
				//	std::cout << "blue: " << (int)blue.at<uint8_t>(i) << " green: " << (int)green.at<uint8_t>(i) << std::endl;
					for (auto & c: connections) {
						if (segments[c.b]->mask.at<uint8_t>(i))
							std::cout << "<<<<<<<<<<<<<<<<<<< " << c.b << " <<<<<<<<<<<<<<<<<<<<<" << std::endl;
					}
				}
			}
			std::cout << std::endl;
			cv::Mat visualization;
			cv::merge({blue, green, red}, visualization);
			cv::imshow("SelectionLikelihood", visualization);
			cv::waitKey();
#endif

			std::uniform_real_distribution<float> dis(0.f, sum);
			float rnd = dis(gen_);

			sum = 0.f;
			for (auto & c: connections) {
				sum += c.similarity;
				if (sum >= rnd) {
					s = s->merge(segments[c.b].get());
					break;
				}
			}
		}

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

PyObject * get_bboxes(PyObject * image_, PyObject * seg_, PyObject * edge_, uint8_t flags, PyObject * weights_, uint32_t n, std::string selection_prior, std::string segment_type, float threshold) {
	NDArrayConverter cvt;
	cv::Mat image   = cvt.toMat(image_);
	cv::Mat seg     = cvt.toMat(seg_);
	cv::Mat edge    = cvt.toMat(edge_);
	cv::Mat weights = cvt.toMat(weights_);
	return cvt.toNDArray(get_bboxes_(image, seg, edge, flags, weights, n, selection_prior, segment_type, threshold));
}

static void init_ar() {
	Py_Initialize();
	import_array();
}

BOOST_PYTHON_MODULE(random_selection) {
	init_ar();

	def("get_bboxes", get_bboxes);
}

int main(int argc, char * argv[]) {
	if (argc != 7) {
		std::cout << "Usage: " << argv[0] << " <image> <segmentation> <edge> <iterations> <selection_prior: uniform, location, objectness, objectness-location> <segment_type: uniform, features, objectness>" << std::endl;
		return 0;
	}

	cv::Mat image = cv::imread(argv[1]);
	cv::Mat seg = cv::imread(argv[2], cv::IMREAD_UNCHANGED);
	cv::Mat edge = cv::imread(argv[3], cv::IMREAD_UNCHANGED);
	uint8_t iterations = atoi(argv[4]);
	std::string selection_prior = std::string(argv[5]);
	std::string segment_type = std::string(argv[6]);

	cv::Mat weights(1, 5, CV_32FC1);
	weights.at<float>(0) = -0.00697891f;
	weights.at<float>(1) = 0.f;
	weights.at<float>(2) = 6.65155577f;
	weights.at<float>(3) = 1.16613659f;
	weights.at<float>(4) = -7.59423175f;
	weights = cv::Mat::ones(1, 6, CV_32FC1);

// 	cv::namedWindow("Image", cv::WINDOW_NORMAL);
// 	cv::imshow("Image", (seg == 338) * 255);
// 	cv::waitKey();

// 	cv::namedWindow("Image", cv::WINDOW_NORMAL);
//	while (cv::waitKey() != 'q') {
		std::clock_t begin = std::clock();
		cv::Mat bboxes = get_bboxes_(image, seg, edge, COLOR_SIMILARITY /*| TEXTURE_SIMILARITY*/  | SIZE_SIMILARITY | BBOX_SIMILARITY, weights, iterations, selection_prior, segment_type, 0.85f); 
		std::clock_t end = std::clock();
		double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
		std::cout << "Times passed in seconds: " << elapsed_secs << std::endl;
//	}

// 	for (int i = 0; i < bboxes.rows; i++) {
// 		cv::Point p1(bboxes.at<int>(i,0), bboxes.at<int>(i,1));
// 		cv::Point p2(bboxes.at<int>(i,2), bboxes.at<int>(i,3));
// 		cv::Mat tmp;
// 		image.copyTo(tmp);
// 		cv::rectangle(tmp, p1, p2, cv::Scalar(255, 255, 0));
// 		cv::imshow("Image", tmp);
// 		cv::waitKey();
// 	}
	return 0;
}
