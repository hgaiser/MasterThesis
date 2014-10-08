//#define DEBUG 1

//#include "CVBoostConverter.hpp"
#include <random>
#include "conversion.h"
#include <boost/python.hpp>

#include "connection.h"
#include "adjacency.h"
#include "uniform_segment.h"
#include "location_prior.h"
#include "random_stopping_criterion.h"

using namespace boost::python;

typedef std::vector<std::shared_ptr<Segment>> LevelSegments;

std::random_device rd_;
std::mt19937 gen_(rd_());

cv::Mat get_bboxes_(const std::vector<cv::Mat> & levels, int n) {
#ifdef DEBUG
	cv::namedWindow("Image", cv::WINDOW_NORMAL);
	for (const auto & lev: levels) {
		cv::Mat draw;
		cv::normalize(lev, draw, 0, 255, cv::NORM_MINMAX, CV_8UC1);
		cv::imshow("Image", draw);
		cv::waitKey();
	}
#endif
	cv::Size size = levels[0].size();

	std::vector<LevelSegments> level_segments;
	level_segments.resize(levels.size());
	cv::Mat bboxes;
	int ind = 0;
	for (int lev = 0; lev < levels.size(); lev++) {
		const cv::Mat & seg = levels[lev];
		double max_id_;
		cv::minMaxIdx(seg, nullptr, &max_id_);
		int max_id = max_id_;
		level_segments[lev].reserve(max_id);
		for (int i = 0; i <= max_id; i++)
			level_segments[lev].push_back(std::make_shared<UniformSegment>(i, size));

		{
			AdjacencyMatrix adjacency(max_id + 1);
			for (int i = 0; i < size.height; i++) {
				for (int j = 0; j < size.width; j++) {
					cv::Point p(j, i);
					int id = seg.at<int>(p);
					level_segments[lev][id]->addPoint(cv::Mat(), cv::Mat(), p);

					if (i < size.height - 1) {
						int n = seg.at<int>(i+1, j);
						if (n != id && adjacency.get(id, n) == false) {
							adjacency.get(id, n) = true;
							level_segments[lev][id]->addNeighbour(n);
							level_segments[lev][n]->addNeighbour(id);
						}
					}

					if (j < size.width - 1) {
						int n = seg.at<int>(i, j+1);
						if (n != id && adjacency.get(id, n) == false) {
							adjacency.get(id, n) = true;
							level_segments[lev][id]->addNeighbour(n);
							level_segments[lev][n]->addNeighbour(id);
						}
					}
				}
			}
		}

		float similarity_sum = 0.f;
		for (auto & s: level_segments[lev]) {
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
	}

	LocationPrior lp;
	std::vector<SelectionPriorMap> prior;
	int segments_sum = 0;
	for (int i = 0; i < level_segments.size(); i++) {
		segments_sum += level_segments[i].size();
		prior.push_back(lp.computeSelectionPrior(levels[i], level_segments[i]));
	}

	for (int i = 0; i < n; i++) {
		int sid = rand() % segments_sum;

		for (int j = 0; j < level_segments.size(); j++) {
			const LevelSegments & lev = level_segments[j];
			const cv::Mat & seg = levels[j];
			if (sid >= lev.size()) {
				sid -= lev.size();
				continue;
			}

			std::shared_ptr<Segment> s = lev[prior[j].poll()];
			RandomStoppingCriterion stop;

			while (s->neighbours.size() && stop.stop(seg, s->bbox()) == false) {
#ifdef DEBUG
				cv::Mat red;
				cv::normalize(seg, red, 0, 255, cv::NORM_MINMAX, CV_8UC1);
				cv::Mat green;
				s->mask.copyTo(green);
				green *= 255;
#endif
				float sum = s->neighbours.size();
				std::vector<Connection> connections;
				//std::cout << "neighbours:" << std::endl;
				for (auto n: s->neighbours) {
					connections.push_back(Connection(s->id, n, 0.f));
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
					blue += level_segments[j][c.b]->mask * 255 * (c.similarity / max_sim);
				}

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
						s = s->merge(level_segments[j][c.b].get());
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
			break;
		}
	}

	return bboxes;
}

PyObject * get_bboxes(PyObject * levels_, int n) {
	NDArrayConverter cvt;
	std::vector<cv::Mat> levels;

	int nrlevels = PyArray_Size(levels_);
	PyObject ** data = (PyObject **)((PyArrayObject *)levels_)->data;
	for (int i = 0; i < nrlevels; i++)
		levels.push_back(cvt.toMat(data[i]));

	return cvt.toNDArray(get_bboxes_(levels, n));
}

static void init_ar() {
	Py_Initialize();
	import_array();
}

BOOST_PYTHON_MODULE(level_selection) {
	init_ar();

	def("get_bboxes", get_bboxes);
}

/*int main(int argc, char * argv[]) {

	return 0;
}*/
