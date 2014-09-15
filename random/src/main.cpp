//#include "CVBoostConverter.hpp"
#include <random>
#include "conversion.h"
#include <boost/python.hpp>

#include "connection.h"
#include "adjacency.h"
#include "uniform_prior.h"
#include "location_prior.h"
#include "objectness_prior.h"
#include "objectness_location_prior.h"
#include "random_stopping_criterion.h"

using namespace boost::python;

//#define DEBUG 1

std::random_device rd_;
std::mt19937 gen_(rd_());

cv::Mat get_bboxes_(cv::Mat image, cv::Mat seg, cv::Mat edge, uint8_t flags, uint32_t n, std::string selection_prior) {
	double max_id_;
	cv::minMaxIdx(seg, nullptr, &max_id_);
	int max_id = max_id_;

	std::vector<Segment> segments;
	segments.reserve(max_id);
	int nchannels = image.channels();
	cv::Size size = image.size();
	for (int i = 0; i <= max_id; i++) {
		segments.push_back(Segment(i, nchannels, size));
	}

	{
		AdjacencyMatrix adjacency(max_id + 1);
		for (int i = 0; i < image.rows; i++) {
			for (int j = 0; j < image.cols; j++) {
				cv::Point p(j, i);
				uint16_t id = seg.at<uint16_t>(p);
				if (nchannels == 1)
					segments[id].addPoint(image.at<cv::Vec<uint8_t, 1>>(p), edge.at<uint8_t>(p), p);
				else if (nchannels == 3)
					segments[id].addPoint(image.at<cv::Vec3b>(p), edge.at<uint8_t>(p), p);

				if (i < image.rows - 1) {
					uint16_t n = seg.at<uint16_t>(i+1, j);
					if (n != id && adjacency.get(id, n) == false) {
						adjacency.get(id, n) = true;
						segments[id].addNeighbour(n);
						segments[n].addNeighbour(id);
					}
				}

				if (j < image.cols - 1) {
					uint16_t n = seg.at<uint16_t>(i, j+1);
					if (n != id && adjacency.get(id, n) == false) {
						adjacency.get(id, n) = true;
						segments[id].addNeighbour(n);
						segments[n].addNeighbour(id);
					}
				}
			}
		}
	}

	cv::Mat bboxes;
	float similarity_sum = 0.f;
	for (auto & s: segments) {
		s.normalizeHistogram();

		cv::Mat bbox = cv::Mat(1, 4, CV_32SC1);
		bbox.at<int>(0) = s.min_p.x;
		bbox.at<int>(1) = s.min_p.y;
		bbox.at<int>(2) = s.max_p.x;
		bbox.at<int>(3) = s.max_p.y;
		if (bboxes.empty())
			bboxes = bbox;
		else
			cv::vconcat(bboxes, bbox, bboxes);
	}

	SelectionPriorMap prior;

	if (selection_prior == "uniform") {
		UniformPrior up;
		prior = up.computeSelectionPrior(image, segments);
	}
	else if (selection_prior == "location") {
		LocationPrior lp;
		prior = lp.computeSelectionPrior(image, segments);
	}
	else if (selection_prior == "objectness") {
		ObjectnessPrior op;
		prior = op.computeSelectionPrior(image, segments);
	}
	else if (selection_prior == "objectness-location") {
		ObjectnessLocationPrior olp;
		prior = olp.computeSelectionPrior(image, segments);
	}
	else {
		throw std::runtime_error("Unknown selection prior method: " + selection_prior);
	}

	Segment s;
	for (uint32_t i = 0; i < n; i++) {
		s = segments[prior.poll()];
		RandomStoppingCriterion stop;
		cv::Rect r(s.min_p, s.max_p);

		while (stop.stop(image, r) == false) {
			float sum = 0.f;
			std::vector<Connection> connections;
			for (auto n: s.neighbours) {
				connections.push_back(Connection(s, segments[n], flags));
				sum += (connections.end() - 1)->similarity;
			}

			std::uniform_real_distribution<float> dis(0.f, sum);
			float rnd = dis(gen_);

			sum = 0.f;
			for (auto & c: connections) {
				sum += c.similarity;
				if (sum >= rnd) {
					s = Segment::merge(s, segments[c.b]);
					break;
				}
			}
		}

		cv::Mat bbox = cv::Mat(1, 4, CV_32SC1);
		bbox.at<int>(0) = s.min_p.x;
		bbox.at<int>(1) = s.min_p.y;
		bbox.at<int>(2) = s.max_p.x;
		bbox.at<int>(3) = s.max_p.y;
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
	cv::addWeighted(draw, 0.5, s.mask * 255, 0.5, 0.0, draw);
	cv::imshow("Image", draw);

	std::cout << bboxes << std::endl;
#endif

	return bboxes;
}

PyObject * get_bboxes(PyObject * image_, PyObject * seg_, PyObject * edge_, uint8_t flags, uint32_t n, std::string selection_prior) {
	NDArrayConverter cvt;
	cv::Mat image, seg, edge;
	image = cvt.toMat(image_);
	seg = cvt.toMat(seg_);
	edge = cvt.toMat(edge_);
	return cvt.toNDArray(get_bboxes_(image, seg, edge, flags, n, selection_prior));
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
	if (argc != 6) {
		std::cout << "Usage: " << argv[0] << " <image> <segmentation> <edge> <iterations> <selection_prior: uniform, location, objectness, objectness-location>" << std::endl;
		return 0;
	}

	cv::Mat image = cv::imread(argv[1]);
	cv::Mat seg = cv::imread(argv[2], cv::IMREAD_UNCHANGED);
	cv::Mat edge = cv::imread(argv[3], cv::IMREAD_UNCHANGED);
	uint8_t iterations = atoi(argv[4]);
	std::string selection_prior = std::string(argv[5]);

// 	cv::namedWindow("Image", cv::WINDOW_NORMAL);
// 	cv::imshow("Image", (seg == 338) * 255);
// 	cv::waitKey();

// 	cv::namedWindow("Image", cv::WINDOW_NORMAL);
//	while (cv::waitKey() != 'q') {
		std::clock_t begin = std::clock();
		cv::Mat bboxes = get_bboxes_(image, seg, edge, COLOR_SIMILARITY | TEXTURE_SIMILARITY/* | SIZE_SIMILARITY | BBOX_SIMILARITY*/, iterations, "objectness"); 
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
