//#define DEBUG 1

//#include "CVBoostConverter.hpp"
#include <random>
#include <stdint.h>
#include <functional>
#include "conversion.h"
#include <boost/python.hpp>

#include "segment.h"
#include "connection.h"
#include "adjacency.h"

using namespace boost::python;

cv::Mat get_bboxes_(const cv::Mat & seg, const cv::Mat & edge) {
	double max_id_;
	cv::minMaxIdx(seg, nullptr, &max_id_);
	int max_id = max_id_;

	std::vector<std::shared_ptr<Segment>> segments;
	std::vector<Connection> connections;
	segments.reserve(max_id);
	cv::Size size = seg.size();
	for (int i = 0; i <= max_id; i++) {
		segments.push_back(std::make_shared<Segment>(i, size));
	}

	{
		//AdjacencyMatrix adjacency(max_id + 1);
		for (int i = 0; i < seg.rows; i++) {
			for (int j = 0; j < seg.cols; j++) {
				cv::Point p(j, i);
				uint16_t id = seg.at<uint16_t>(p);
				segments[id]->addPoint(p);

				if (i < seg.rows - 1) {
					uint16_t n = seg.at<uint16_t>(i+1, j);
					if (n != id) {// && adjacency.get(id, n) == false) {
						//adjacency.get(id, n) = true;
						segments[id]->addNeighbour(n, edge.at<uint8_t>(i+1, j));
						segments[n]->addNeighbour(id, edge.at<uint8_t>(i+1, j));
					}
				}

				if (j < seg.cols - 1) {
					uint16_t n = seg.at<uint16_t>(i, j+1);
					if (n != id) { // && adjacency.get(id, n) == false) {
						//adjacency.get(id, n) = true;
						segments[id]->addNeighbour(n, edge.at<uint8_t>(i, j+1));
						segments[n]->addNeighbour(id, edge.at<uint8_t>(i, j+1));
					}
				}
			}
		}
	}

	cv::Mat bboxes;
	{
		AdjacencyMatrix adjacency(max_id + 1);
		for (auto & s: segments) {
			if (s->empty())
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

			for (auto & n: s->neighbours) {
				if (adjacency.get(s->id, n.first) == false) {
					adjacency.get(s->id, n.first) = true;
					connections.push_back(Connection(s->id, segments[n.first]->id, s->computeSimilarity(segments[n.first].get())));
				}
			}
		}
	}

#ifdef DEBUG
	cv::namedWindow("Segment", cv::WINDOW_NORMAL);
#endif

	while (connections.size() != 0) {
		std::sort(connections.begin(), connections.end());
		Connection c = *connections.begin();
		connections.erase(connections.begin());
		std::shared_ptr<Segment> s = segments[c.a]->merge(connections, segments, segments[c.b].get());

#ifdef DEBUG
		cv::Mat draw = cv::Mat::zeros(seg.size(), CV_8UC1);
		draw += segments[c.a]->mask * 127;
		draw += segments[c.b]->mask * 255;
		cv::imshow("Segment", draw);
		cv::waitKey();
#endif

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

	return bboxes;
}

PyObject * get_bboxes(PyObject * seg_, PyObject * edge_) {
	NDArrayConverter cvt;
	cv::Mat seg     = cvt.toMat(seg_);
	cv::Mat edge    = cvt.toMat(edge_);
	return cvt.toNDArray(get_bboxes_(seg, edge));
}

static void init_ar() {
	Py_Initialize();
	import_array();
}

BOOST_PYTHON_MODULE(gpb_selection) {
	init_ar();

	def("get_bboxes", get_bboxes);
}

int main(int argc, char * argv[]) {
	if (argc != 3) {
		std::cout << "Usage: " << argv[0] << " <segmentation> <edge>" << std::endl;
		return 0;
	}

	cv::Mat seg = cv::imread(argv[1], cv::IMREAD_UNCHANGED);
	cv::Mat edge = cv::imread(argv[2], cv::IMREAD_UNCHANGED);

// 	cv::namedWindow("Image", cv::WINDOW_NORMAL);
// 	cv::imshow("Image", (seg == 338) * 255);
// 	cv::waitKey();

// 	cv::namedWindow("Image", cv::WINDOW_NORMAL);
//	while (cv::waitKey() != 'q') {
		std::clock_t begin = std::clock();
		cv::Mat bboxes = get_bboxes_(seg, edge); 
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
