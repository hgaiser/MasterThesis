//#define DEBUG 1

//#include "CVBoostConverter.hpp"
#include <random>
#include <unordered_set>
#include <stdint.h>
#include <functional>
#include "conversion.h"
#include <boost/python.hpp>

#include "uniform_segment.h"

#include "connection.h"
#include "adjacency.h"

using namespace boost::python;


std::random_device rd_;
std::mt19937 gen_(rd_());

float jaccardSimilarity(cv::Rect bbox1, cv::Rect bbox2) {
	cv::Vec4i bi(std::max(bbox1.x, bbox2.x), std::max(bbox1.y,bbox2.y), std::min(bbox1.br().x,bbox2.br().x), std::min(bbox1.br().y,bbox2.br().y));
	int iw = bi[2] - bi[0];
	int ih = bi[3] - bi[1];
	if (iw > 0 && ih > 0) {
		int un = (bbox1.br().x - bbox1.x) * (bbox1.br().y - bbox1.y) +
		(bbox2.br().x - bbox2.x) * (bbox2.br().y - bbox2.y) - iw * ih;
		return iw * ih / float(un);
	}

	return 0.f;
}

cv::Mat get_bboxes_(const cv::Mat & image, const cv::Mat & seg, int x1, int y1, int x2, int y2) {
	double max_id_;
	cv::minMaxIdx(seg, nullptr, &max_id_);
	int max_id = max_id_;

	std::vector<std::shared_ptr<Segment>> segments;
	segments.reserve(max_id);
	int nchannels = image.channels();
	cv::Size size = image.size();
	for (int i = 0; i <= max_id; i++) {
		segments.push_back(std::make_shared<UniformSegment>(i, size));
	}

	{
		AdjacencyMatrix adjacency(max_id + 1);
		for (int i = 0; i < image.rows; i++) {
			for (int j = 0; j < image.cols; j++) {
				cv::Point p(j, i);
				uint16_t id = seg.at<uint16_t>(p);
				segments[id]->addPoint(image, p);

				if (i < image.rows - 1) {
					uint16_t n = seg.at<uint16_t>(i+1, j);
					if (n != id && adjacency.get(id, n) == false) {
						adjacency.get(id, n) = true;
						segments[id]->addNeighbour(n);
						segments[n]->addNeighbour(id);
					}
				}

				if (j < image.cols - 1) {
					uint16_t n = seg.at<uint16_t>(i, j+1);
					if (n != id && adjacency.get(id, n) == false) {
						adjacency.get(id, n) = true;
						segments[id]->addNeighbour(n);
						segments[n]->addNeighbour(id);
					}
				}
			}
		}
	}

	cv::Mat bboxes_out;
	float similarity_sum = 0.f;
	for (auto & s: segments) {
		s->finalizeSetup();
	}

	//for (uint32_t i = 0; i < bboxes.rows; i++) {
		//cv::Rect truth(cv::Point(bboxes.at<int>(i, 0), bboxes.at<int>(i, 1)), cv::Point(bboxes.at<int>(i, 2), bboxes.at<int>(i, 3)));
		cv::Rect truth(cv::Point(x1, y1), cv::Point(x2, y2));
		cv::Mat mask = cv::Mat::zeros(image.size(), CV_8UC1);
		cv::rectangle(mask, truth, cv::Scalar(1), CV_FILLED);

		std::unordered_set<uint32_t> contained_segments;
		std::unordered_set<uint32_t> crossing_segments;
		for (const auto & s: segments) {
			cv::Mat masked;
			mask.copyTo(masked, s->mask);
			int count = cv::countNonZero(masked);
			if (count == s->size)
				contained_segments.insert(s->id);
			else if (count)
				crossing_segments.insert(s->id);
		}

		std::shared_ptr<Segment> s = std::make_shared<UniformSegment>(-1, image.size());
		if (contained_segments.size()) {
			for (const int & id: contained_segments) {
				s = s->merge(segments[id].get());
			}
		} else {
			s = s->merge(segments[seg.at<uint16_t>(truth.y + truth.height / 2, truth.x + truth.width)].get());
		}

		std::cout << "size: " << crossing_segments.size() << std::endl;

#ifdef DEBUG
		cv::Mat draw;
		s->mask.copyTo(draw);
		draw *= 255;
		/*for (const int & n: s->neighbours) {
			draw += segments[n]->mask * 127;
		}*/
		//cv::rectangle(draw, cv::Point(bboxes.at<int>(0, 0), bboxes.at<int>(0, 1)), cv::Point(bboxes.at<int>(0, 2), bboxes.at<int>(0, 3)), cv::Scalar(255));
		cv::namedWindow("Mask", cv::WINDOW_NORMAL);
		cv::imshow("Mask", draw);
		cv::waitKey();
#endif

		float max_sim = jaccardSimilarity(truth, cv::Rect(s->min_p, s->max_p));
#ifdef DEBUG
		std::cout << "max_sim: " << max_sim << std::endl;
#endif
		bool quit = false;
		while (quit == false) {
			quit = true;
			for (const int & n: s->neighbours) {
				std::shared_ptr<Segment> s2 = s->merge(segments[n].get());
				float sim = jaccardSimilarity(truth, cv::Rect(s2->min_p, s2->max_p));
				if (sim > max_sim) {
					s = s2;
					max_sim = sim;
					quit = false;
#ifdef DEBUG
					std::cout << "new max sim: " << max_sim << std::endl;
					cv::imshow("Mask", s->mask * 255);
					cv::waitKey();
#endif
					break;
				}
			}
		}

		cv::Mat bbox = cv::Mat(1, 4, CV_32SC1);
		bbox.at<int>(0) = s->min_p.x;
		bbox.at<int>(1) = s->min_p.y;
		bbox.at<int>(2) = s->max_p.x;
		bbox.at<int>(3) = s->max_p.y;
		if (bboxes_out.empty())
			bboxes_out = bbox;
		else
			cv::vconcat(bboxes_out, bbox, bboxes_out);
	//}

#ifdef DEBUG
	std::cout << "bboxes: " << bboxes_out << std::endl;
#endif

	return bboxes_out;
}

PyObject * get_bboxes(PyObject * image_, PyObject * seg_, int x1, int y1, int x2, int y2) {
	NDArrayConverter cvt;
	cv::Mat image  = cvt.toMat(image_);
	cv::Mat seg    = cvt.toMat(seg_);
	//cv::Mat bboxes = cvt.toMat(bboxes_);
	std::cout << x1 << " " << y1 << " " << x2 << " " << y2 << std::endl;
	return cvt.toNDArray(get_bboxes_(image, seg, x1, y1, x2, y2));
}

static void init_ar() {
	Py_Initialize();
	import_array();
}

BOOST_PYTHON_MODULE(brute_selection) {
	init_ar();

	def("get_bboxes", get_bboxes);
}

int main(int argc, char * argv[]) {
	if (argc != 7) {
		std::cout << "Usage: " << argv[0] << " <image> <segmentation>" << std::endl;
		return 0;
	}

	cv::Mat image = cv::imread(argv[1]);
	cv::Mat seg = cv::imread(argv[2], cv::IMREAD_UNCHANGED);
	cv::Mat bboxes = cv::Mat::zeros(1, 4, CV_32SC1);
	bboxes.at<int>(0) = 220;
	bboxes.at<int>(1) = 200;
	bboxes.at<int>(2) = 350;
	bboxes.at<int>(3) = 390;

// 	cv::namedWindow("Image", cv::WINDOW_NORMAL);
// 	cv::rectangle(image, cv::Point(220, 200), cv::Point(350, 390), cv::Scalar(255, 255, 0));
// 	cv::imshow("Image", image);
// 	cv::waitKey();

// 	cv::namedWindow("Image", cv::WINDOW_NORMAL);
//	while (cv::waitKey() != 'q') {
		std::clock_t begin = std::clock();
		cv::Mat bboxes_result = get_bboxes_(image, seg, atoi(argv[3]), atoi(argv[4]), atoi(argv[5]), atoi(argv[6]));
		//cv::Mat bboxes_result = get_bboxes_(image, seg, bboxes);
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
