#define DEBUG 1

#include "CVBoostConverter.hpp"
#include <boost/python.hpp>
#include <opencv2/opencv.hpp>
#include <unordered_map>
#include "edge.h"

using namespace boost::python;


class AdjacencyMatrix : public std::vector<Edge> {
public:
	AdjacencyMatrix(int n) {
		this->n = n;
		resize((n + 1)*n/2);
	}

	Edge& get(int i, int j) {
		int tmp = i > j ? j : i;
		j = i > j ? i : j;
		i = tmp;
		int ind = index(i, j);
		return operator[](ind);
	}

protected:
	int n;

	int index(int i, int j) {
		return (n * i) + j - ((i * (i+1)) / 2); 
	}
};

void updateSurface(cv::Point p, int s, std::unordered_map<int, std::pair<cv::Point, cv::Point>> & surfaces, cv::Size size) {
	cv::Point p1, p2;
	auto it = surfaces.find(s);
	if (it != surfaces.end()) {
		p1 = it->second.first;
		p2 = it->second.second;
	}
	else {
		p1 = cv::Point(size.width, size.height);
		p2 = cv::Point(0, 0);
	}
	cv::Point minp = cv::Point(p1.x < p.x ? p1.x : p.x, p1.y < p.y ? p1.y : p.y);
	cv::Point maxp = cv::Point(p2.x > p.x ? p2.x : p.x, p2.y > p.y ? p2.y : p.y);
	surfaces[s] = {minp, maxp};
}

cv::Mat makeTree(cv::Mat s, cv::Mat e) {
	if (s.type() != CV_32SC1) {
		std::cerr << "Input matrix should be 32 bit integer." << std::endl;
		return cv::Mat::zeros(10,10,CV_8UC1);
	}

	double n = 0;
	cv::minMaxLoc(s, nullptr, &n);
	cv::Mat edges = cv::Mat::zeros(s.size(), CV_8UC1);
	AdjacencyMatrix adjacency(n);
	std::unordered_map<int, std::pair<cv::Point, cv::Point>> surfaces;

	cv::Mat nonzero;
	cv::findNonZero(s == -1, nonzero);
	for (int i = 0; i < nonzero.total(); i++) {
		cv::Point p = nonzero.at<cv::Point>(i);

		if (s.at<int>(p.y-1, p.x) > 0 && s.at<int>(p.y+1, p.x) > 0 && 
			s.at<int>(p.y-1, p.x) != s.at<int>(p.y+1, p.x)) {
			edges.at<uint8_t>(p.y, p.x) = 255;

			int a = s.at<int>(p.y-1, p.x);
			int b = s.at<int>(p.y+1, p.x);

			updateSurface(p, a, surfaces, s.size());
			updateSurface(p, b, surfaces, s.size());

			Edge & c = adjacency.get(a, b);
			c.edges.push_back(e.at<uint8_t>(p));
			c.updateBox(p);
		}
		else if (s.at<int>(p.y, p.x - 1) > 0 && s.at<int>(p.y, p.x + 1) > 0 && 
			s.at<int>(p.y, p.x - 1) != s.at<int>(p.y, p.x + 1)) {
			edges.at<uint8_t>(p.y, p.x) = 255;

			int a = s.at<int>(p.y, p.x-1);
			int b = s.at<int>(p.y, p.x+1);

			updateSurface(p, a, surfaces, s.size());
			updateSurface(p, b, surfaces, s.size());

			Edge & c = adjacency.get(a, b);
			c.edges.push_back(e.at<uint8_t>(p));
			c.updateBox(p);
		}
	}

	int max_id = 0;
	std::multiset<Edge> connections;
	for (int i = 1; i < n; i++) {
		for (int j = i + 1; j < n; j++) {
			Edge & c = adjacency.get(i, j);
			if (c.edges.size()) {
				c.a = i;
				c.b = j;

#ifdef DEBUG
				c.a_mask = s == c.a;
				c.b_mask = s == c.b;
#endif

				c.calcMedian();
				connections.insert(c);
				max_id = i > max_id ? i : max_id;
				max_id = j > max_id ? j : max_id;
			}
		}
	}

	/*for (auto it = connections.begin(); it != connections.end(); ++it) {
		std::cout << "connection: a: " << it->a << " b: " << it->b << " median: " << it->median << std::endl;
	}*/

	while (connections.size() != 0) {
		Edge removal = *connections.begin();

		connections.erase(connections.begin());
		max_id++;

		cv::Point minp1 = surfaces[removal.a].first;
		cv::Point maxp1 = surfaces[removal.a].second;
		cv::Point minp2 = surfaces[removal.b].first;
		cv::Point maxp2 = surfaces[removal.b].second;
		surfaces[max_id] = {cv::Point(minp1.x < minp2.x ? minp1.x : minp2.x, minp1.y < minp2.y ? minp1.y : minp2.y), 
							cv::Point(maxp1.x > maxp2.x ? maxp1.x : maxp2.x, maxp1.y > maxp2.y ? maxp1.y : maxp2.y)};
		
#ifdef DEBUG
		std::cout << "removing: a: " << removal.a << " b: " << removal.b << " median: " << removal.median << std::endl;
		for (auto edge: removal.edges)
			std::cout << int(edge) << ", ";
		std::cout << std::endl;

		cv::Mat mask = removal.a_mask + removal.b_mask;
		std::cout << "merging <" << removal.a << ", " << removal.b << ">; surface: <" << surfaces[max_id].first << ", " << surfaces[max_id].second << ">" << std::endl;
		cv::Mat draw, rgb_mask;
		cv::cvtColor(e, draw, cv::COLOR_GRAY2RGB);
		cv::cvtColor(mask, rgb_mask, cv::COLOR_GRAY2RGB);
		cv::addWeighted(draw, 0.5, rgb_mask, 0.5, 0, draw);
		cv::rectangle(draw, surfaces[max_id].first, surfaces[max_id].second, cv::Scalar(255, 0, 0));
		cv::imshow("Edge", draw);
		cv::waitKey();
#endif

		if (connections.size() == 0)
			break;

		std::vector<Edge> add_list;
		std::vector<std::multiset<Edge>::iterator> remove_list;
		std::unordered_map<int, Edge> neighbours;

		for (auto it = connections.begin(); it != connections.end(); ++it) {
			bool conn_a = it->a == removal.a || it->a == removal.b;
			bool conn_b = it->b == removal.a || it->b == removal.b;

			if (conn_a || conn_b) {
				remove_list.push_back(it);

				Edge c = *it;
				if (conn_a) {
					c.a = max_id;
#ifdef DEBUG
					c.a_mask = mask;
#endif
				}
				if (conn_b) {
					c.b = max_id;
#ifdef DEBUG
					c.b_mask = mask;
#endif
				}

				int nid = conn_a ? c.b : c.a;
				auto nit = neighbours.find(nid);
				if (nit == neighbours.end()) {
					neighbours.insert({nid, c});
				} else {
					std::vector<uint8_t> edges;
					edges.reserve(c.edges.size() + nit->second.edges.size());
					std::merge(c.edges.begin(), c.edges.end(), nit->second.edges.begin(), nit->second.edges.end(), std::back_inserter(edges));

					neighbours.erase(nit);
					c.edges = edges;
					c.calcMedian();
					add_list.push_back(c);
				}
			}
		}

		for (auto it = remove_list.rbegin(); it != remove_list.rend(); ++it)
			connections.erase(*it);
		for (auto c: add_list)
			connections.insert(c);
		for (auto n: neighbours)
			connections.insert(n.second);
	}

	cv::Mat result(surfaces.size(), 4, CV_32SC1);
	int i = 0;
	for (auto sur: surfaces) {
		result.at<cv::Point>(i++) = sur.second.first;
		result.at<cv::Point>(i++) = sur.second.second;
	}

	return result;
}

static void init_ar()
{
    Py_Initialize();
    import_array();
}

BOOST_PYTHON_MODULE(segmentation_tree)
{
    init_ar();

	//initialize converters
	to_python_converter<cv::Mat,
		bcvt::matToNDArrayBoostConverter>();
		bcvt::matFromNDArrayBoostConverter();

    def("tree", makeTree);
}

int main(int argc, char * argv[]) {
	if (argc != 3) {
		std::cout << "Please use as ./seg_tree <path to segmentation> <path to edge map>" << std::endl;
		return 0;
	}
	cv::Mat segmentation = cv::imread(argv[1], cv::IMREAD_UNCHANGED);
	segmentation.convertTo(segmentation, CV_32SC1);
	cv::Mat edge = cv::imread(argv[2], cv::IMREAD_UNCHANGED);

	cv::Mat nonzero;
	cv::findNonZero(segmentation == (1 << 16) - 1, nonzero);
	for (int i = 0; i < nonzero.total(); i++) {
		cv::Point p = nonzero.at<cv::Point>(i);
		segmentation.at<int>(p) = -1;
	}

	cv::namedWindow("Segmentation", cv::WINDOW_NORMAL);
	cv::namedWindow("Edge", cv::WINDOW_NORMAL);

	cv::imshow("Segmentation", segmentation);
	cv::imshow("Edge", edge);

	std::clock_t begin = std::clock();
	cv::Mat tree = makeTree(segmentation, edge);
	std::clock_t end = std::clock();
	std::cout << tree.rows << " surfaces detected." << std::endl;
	double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
	std::cout << "Times passed in seconds: " << elapsed_secs << std::endl;

	cv::waitKey();
	return 0;
}
