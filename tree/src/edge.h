#ifndef EDGE_H_
#define EDGE_H_

class Edge {
public:
	Edge() : a(0), b(0), median(0) {}
	Edge(int a, int b, std::vector<uint8_t> edges) : 
		a(a), b(b), edges(edges) {}
	Edge(int a, int b, std::vector<uint8_t> edges, bool compute) : 
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

	bool operator<(const Edge & other) const {
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

#endif

