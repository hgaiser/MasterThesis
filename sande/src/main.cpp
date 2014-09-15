//#include "CVBoostConverter.hpp"
#include <random>
#include "conversion.h"
#include <boost/python.hpp>

#include "connection.h"
#include "adjacency.h"

using namespace boost::python;

std::random_device rd_;
std::mt19937 gen_(rd_());

//#define DEBUG 1

cv::Mat get_bboxes_(cv::Mat image, cv::Mat seg, cv::Mat edge, uint8_t flags, uint8_t iterations) {
	double max_id_;
	cv::minMaxIdx(seg, nullptr, &max_id_);
	int max_id = max_id_;

	std::vector<Segment> segments;
	std::vector<Connection> connections;
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

	float similarity_sum = 0.f;
	{
		AdjacencyMatrix adjacency(max_id + 1);
		for (auto & s: segments) {
			s.normalizeHistogram();
			for (auto n: s.neighbours) {
				if (adjacency.get(s.id, n) == false) {
					adjacency.get(s.id, n) = true;
					Connection c(s, segments[n], flags);
					connections.push_back(c);
					similarity_sum += c.similarity;
				}
			}
		}
	}

	std::vector<Segment> final_segments(segments);
	bool stochastic = iterations > 1;
	for (uint8_t i = 0; i < iterations; i++) {
		std::vector<Connection> connections_(connections);
		std::vector<Segment> segments_(segments);
		float similarity_sum_ = similarity_sum;
		while (connections_.size() != 0) {
			std::uniform_real_distribution<float> dis(0.f, similarity_sum_);
			std::sort(connections_.begin(), connections_.end());
			float sum = 0.f;
			float rnd = dis(gen_);
			for (auto it = connections_.begin(); it != connections_.end(); it++) {
				sum += it->similarity;
				if (sum >= rnd || stochastic == false) {
					similarity_sum_ -= it->similarity;
					Connection c = *it;
					connections_.erase(it);
					c.merge(connections_, segments_, flags, similarity_sum_);
					break;
				}
			}
		}

		final_segments.insert(final_segments.end(), segments_.begin() + segments.size(), segments_.end());
	}

	cv::Mat bboxes;
	for (auto s: final_segments) {
		if (s.size == 0)
			continue;

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
	cv::Mat gray;
	cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
	cv::namedWindow("Segment", cv::WINDOW_NORMAL);
	for (auto s: final_segments) {
		cv::Mat display;
		cv::addWeighted(s.mask * 255, 0.5, gray, 0.5, 0.0, display);
		cv::rectangle(display, s.min_p, s.max_p, cv::Scalar(255));
		cv::imshow("Segment", display);
		cv::waitKey();
	}
#endif

	return bboxes;
}

PyObject * get_bboxes(PyObject * image_, PyObject * seg_, PyObject * edge_, uint8_t flags, uint8_t iterations) {
	NDArrayConverter cvt;
	cv::Mat image, seg, edge;
	image = cvt.toMat(image_);
	seg = cvt.toMat(seg_);
	edge = cvt.toMat(edge_);
	return cvt.toNDArray(get_bboxes_(image, seg, edge, flags, iterations));
}

static void init_ar() {
	Py_Initialize();
	import_array();
}

BOOST_PYTHON_MODULE(selective_search) {
	init_ar();

	//initialize converters
	/*to_python_converter<cv::Mat,
		bcvt::matToNDArrayBoostConverter>();
		bcvt::matFromNDArrayBoostConverter();*/

	def("get_bboxes", get_bboxes);
}

int main(int argc, char * argv[]) {
	if (argc != 5) {
		std::cout << "Usage: " << argv[0] << " <image> <segmentation> <edge> <iterations>" << std::endl;
		return 0;
	}

	cv::Mat image = cv::imread(argv[1]);
	cv::Mat seg = cv::imread(argv[2], cv::IMREAD_UNCHANGED);
	cv::Mat edge = cv::imread(argv[3], cv::IMREAD_UNCHANGED);
	uint8_t iterations = atoi(argv[4]);

// 	cv::namedWindow("Image", cv::WINDOW_NORMAL);
// 	cv::imshow("Image", (seg == 338) * 255);
// 	cv::waitKey();

	std::clock_t begin = std::clock();
	cv::Mat bboxes = get_bboxes_(image, seg, edge, COLOR_SIMILARITY | TEXTURE_SIMILARITY | SIZE_SIMILARITY | BBOX_SIMILARITY, iterations); 
	std::clock_t end = std::clock();
	double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
	std::cout << "Times passed in seconds: " << elapsed_secs << std::endl;

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
