#include <opencv2/opencv.hpp>
#include <unordered_map>

//#define DEBUG 1

class Connection {
public:
	Connection() : a(0), b(0), median(0) {}
	Connection(int a, int b, std::vector<uint8_t> edges) : 
		a(a), b(b), edges(edges) {}
	Connection(int a, int b, std::vector<uint8_t> edges, bool compute) : 
		a(a), b(b), edges(edges)
	{
		if (compute) calcMedian();
	}

	int calcMedian() {
		std::sort(edges.begin(), edges.end());
		median = edges[edges.size()/2];
		return median;
	}

	void updateBox(cv::Point p) {
		min.x = min.x < p.x ? min.x : p.x;
		min.y = min.y < p.y ? min.y : p.y;
		max.x = max.x > p.x ? max.x : p.x;
		max.y = max.y > p.y ? max.y : p.y;
	}

	cv::Rect getRect() {
		return cv::Rect(min.x, min.x, max.x - min.x, max.y - min.y);
	}

	int median;

    bool operator<(const Connection & other) const {
        return median < other.median;
    }

    std::vector<uint8_t> edges;
    cv::Point min;
    cv::Point max;

	int a;
	int b;

#ifdef DEBUG
	cv::Mat a_mask;
	cv::Mat b_mask;
#endif
};

class AdjacencyMatrix: std::vector<Connection> {
public:
	AdjacencyMatrix(int n) {
		this->n = n;
		resize((n + 1)*n/2);
	}

	Connection& get(int i, int j) {
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

void makeTree(cv::Mat s, cv::Mat e) {
	int n = 255;
	//cv::Mat edges = cv::Mat::zeros(s.size(), CV_8UC1);
	AdjacencyMatrix adjacency(n);
	std::unordered_map<int, std::pair<cv::Point, cv::Point>> surfaces;

	cv::Mat nonzero;
	cv::findNonZero(s == 0, nonzero);
	for (int i = 0; i < nonzero.total(); i++) {
		cv::Point p = nonzero.at<cv::Point>(i);

		if (s.at<uint8_t>(p.y-1, p.x) != 255 && s.at<uint8_t>(p.y+1, p.x) != 255 && 
			s.at<uint8_t>(p.y-1, p.x) != 0 && s.at<uint8_t>(p.y+1, p.x) != 0 && 
			s.at<uint8_t>(p.y-1, p.x) != s.at<uint8_t>(p.y+1, p.x)) {
			//edges.at<uint8_t>(p.y, p.x) = 255;

			int a = s.at<uint8_t>(p.y-1, p.x);
			int b = s.at<uint8_t>(p.y+1, p.x);

			updateSurface(p, a, surfaces, s.size());
			updateSurface(p, b, surfaces, s.size());

			Connection & c = adjacency.get(a, b);
			c.edges.push_back(e.at<uint8_t>(p));
			c.updateBox(p);
		}
		else if (s.at<uint8_t>(p.y, p.x - 1) != 255 && s.at<uint8_t>(p.y, p.x + 1) != 255 && 
			s.at<uint8_t>(p.y, p.x - 1) != 0 && s.at<uint8_t>(p.y, p.x + 1) != 0 && 
			s.at<uint8_t>(p.y, p.x - 1) != s.at<uint8_t>(p.y, p.x + 1)) {
			//edges.at<uint8_t>(p.y, p.x) = 255;

			int a = s.at<uint8_t>(p.y, p.x-1);
			int b = s.at<uint8_t>(p.y, p.x+1);

			updateSurface(p, a, surfaces, s.size());
			updateSurface(p, b, surfaces, s.size());

			Connection & c = adjacency.get(a, b);
			c.edges.push_back(e.at<uint8_t>(p));
			c.updateBox(p);
		}
	}

	int max_id = 0;
	std::multiset<Connection> connections;
	for (int i = 1; i < n; i++) {
		for (int j = i + 1; j < n; j++) {
			Connection & c = adjacency.get(i, j);
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
		Connection removal = *connections.begin();

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

		std::vector<Connection> add_list;
		std::vector<std::multiset<Connection>::iterator> remove_list;
		std::unordered_map<int, Connection> neighbours;

		for (auto it = connections.begin(); it != connections.end(); ++it) {
			bool conn_a = it->a == removal.a || it->a == removal.b;
			bool conn_b = it->b == removal.a || it->b == removal.b;

			if (conn_a || conn_b) {
				remove_list.push_back(it);

				Connection c = *it;
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

	std::cout << "size: " << surfaces.size() << std::endl;

	/*for (auto sur: surfaces) {
		std::cout << "surface: " << sur.first << " rect: <" << sur.second.first << ", " << sur.second.second << ">" << std::endl;
		cv::Mat draw;
		s.copyTo(draw);
		cv::rectangle(draw, sur.second.first, sur.second.second, cv::Scalar(255));
		cv::imshow("Segmentation", draw);
		cv::waitKey();
	}*/
}

int main(int argc, char * argv[]) {
	cv::Mat segmentation = cv::imread("../segmentation.png", cv::IMREAD_GRAYSCALE);
	cv::Mat edge = cv::imread("../edge.png", cv::IMREAD_GRAYSCALE);

	cv::namedWindow("Segmentation", cv::WINDOW_NORMAL);
	cv::namedWindow("Edge", cv::WINDOW_NORMAL);

	cv::imshow("Segmentation", segmentation);
	cv::imshow("Edge", edge);

	std::clock_t begin = std::clock();
	makeTree(segmentation, edge);
	std::clock_t end = std::clock();
	double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
	std::cout << "Times passed in seconds: " << elapsed_secs << std::endl;

	cv::waitKey();
	return 0;
}