#pragma once

#include <unordered_map>
#include <opencv2/opencv.hpp>
#include "connection.h"

class Segment
{
public:
	Segment(int id, cv::Size s) :
#ifdef DEBUG
		mask(cv::Mat::zeros(s, CV_8UC1)),
#endif
		id(id),
		min_p(s.width, s.height),
		max_p(0, 0),
		im_size(s),
		size(0)
	{
	}

	virtual void addPoint(cv::Point point) {
#ifdef DEBUG
		mask.at<uint8_t>(point) = 1;
#endif

		min_p = cv::Point(std::min(point.x, min_p.x), std::min(point.y, min_p.y));
		max_p = cv::Point(std::max(point.x, max_p.x), std::max(point.y, max_p.y));

		size++;
	}

	void addNeighbour(int n, uint8_t texture) {
		if (neighbours.find(n) == neighbours.end())
			neighbours.insert({n, { texture }});
		else
			neighbours[n].push_back(texture);
	}

	float computeSimilarity(const Segment * b_) {
		std::vector<uint8_t> & edge = neighbours[b_->id];
		if (edge.size() == 0) {
			return 0.f;
		}

		std::sort(edge.begin(), edge.end());
		return edge[edge.size() / 2];
	}

	std::shared_ptr<Segment> merge(std::vector<Connection> & connections, std::vector<std::shared_ptr<Segment>> & segments, const Segment * b) {
		std::shared_ptr<Segment> s = std::make_shared<Segment>(segments.size(), im_size);
#ifdef DEBUG
		cv::bitwise_or(mask, b->mask, s->mask);
#endif
		s->min_p = cv::Point(std::min(min_p.x, b->min_p.x), std::min(min_p.y, b->min_p.y));
		s->max_p = cv::Point(std::max(max_p.x, b->max_p.x), std::max(max_p.y, b->max_p.y));

		s->neighbours = neighbours;
		s->neighbours.erase(b->id);
		for (auto & n: b->neighbours) {
			if (n.first == id)
				continue;

			if (s->neighbours.find(n.first) == s->neighbours.end())
				s->neighbours.insert(n);
			else
				s->neighbours[n.first].insert(s->neighbours[n.first].end(), n.second.begin(), n.second.end());
		}

		for (auto it = connections.begin(); it != connections.end(); ) {
			if (it->a == id || it->a == b->id || it->b == id || it->b == b->id) {
				it = connections.erase(it);
			}
			else
				it++;
		}

		for (auto & neighbour : s->neighbours) {
			std::shared_ptr<Segment> & n = segments[neighbour.first];
			auto it = n->neighbours.find(id);
			if (it != n->neighbours.end()) {
				n->neighbours.insert({ s->id, n->neighbours[id] });
				n->neighbours.erase(it);

				if (n->neighbours.find(b->id) != n->neighbours.end()) {
					n->neighbours[s->id].insert(n->neighbours[s->id].end(), n->neighbours[b->id].begin(), n->neighbours[b->id].end());
					n->neighbours.erase(b->id);
				}
			} else if (n->neighbours.find(b->id) != n->neighbours.end()) {
				n->neighbours.insert({ s->id, n->neighbours[b->id] });
				n->neighbours.erase(b->id);
			}

			connections.push_back(Connection(s->id, n->id, s->computeSimilarity(n.get())));
		}

		segments.push_back(s);
// 		std::cout << "connection size: " << connections.size() << std::endl;

		return s;
	}

	std::ostream & output(std::ostream & out) const {
		return out;
	}

	inline cv::Rect bbox() const {
		return cv::Rect(min_p, max_p);
	}

	inline bool empty() {
		return size == 0;
	}

	std::unordered_map<int, std::vector<uint8_t>> neighbours;
#ifdef DEBUG
	cv::Mat mask;
#endif
	int id;
	cv::Point min_p;
	cv::Point max_p;
	cv::Size im_size;
	int size;
};

std::ostream & operator<<(std::ostream & os, const Segment & b) {
    return b.output(os);
}

