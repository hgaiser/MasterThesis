#ifndef CONNECTION_H_
#define CONNECTION_H_

#include "segment.h"

class Connection
{
public:
	Connection(Segment a_, Segment b_, uint8_t flags) :
		a(a_.id), b(b_.id), similarity(exp(Segment::computeSimilarity(a_, b_, flags)))
	{
	}

	void merge(std::vector<Connection> & connections, std::vector<Segment> & segments, uint8_t flags, float & similarity_sum) {
		Segment & a_ = segments[a];
		Segment & b_ = segments[b];

		Segment s(segments.size(), a_.nchannels, a_.im_size_cv);
		s.size = a_.size + b_.size;
		cv::bitwise_or(a_.mask, b_.mask, s.mask);

		if (a_.color_hist.size() != b_.color_hist.size() || a_.texture_hist.size() != b_.texture_hist.size())
			return;

		for (uint64_t i = 0; i < a_.color_hist.size(); i++) {
			s.color_hist[i] = (a_.color_hist[i] * a_.size + b_.size * b_.color_hist[i]) / s.size;
		}

		for (uint64_t i = 0; i < a_.texture_hist.size(); i++)
			s.texture_hist[i] = (a_.size * a_.texture_hist[i] + b_.size * b_.texture_hist[i]) / s.size;
		//s.texture_hist = (a_.size * a_.texture_hist + b_.size * b_.texture_hist) / s.size;

		s.min_p = cv::Point(min(a_.min_p.x, b_.min_p.x), min(a_.min_p.y, b_.min_p.y));
		s.max_p = cv::Point(max(a_.max_p.x, b_.max_p.x), max(a_.max_p.y, b_.max_p.y));

		for (auto n: a_.neighbours) {
			if (n != b)
				s.neighbours.insert(n);
		}
		for (auto n: b_.neighbours) {
			if (n != a)
				s.neighbours.insert(n);
		}

		for (auto it = connections.begin(); it != connections.end(); ) {
			if (it->a == a || it->a == b || it->b == a || it->b == b) {
				similarity_sum -= it->similarity;
				it = connections.erase(it);
			}
			else
				it++;
		}

		for (auto n: s.neighbours) {
			Segment & neighbour = segments[n];
			neighbour.neighbours.erase(a);
			neighbour.neighbours.erase(b);
			neighbour.neighbours.insert(s.id);
			Connection c(s, neighbour, flags);
			connections.push_back(c);
			similarity_sum += c.similarity;
		}

		segments.push_back(s);
	}

	friend std::ostream& operator<<(std::ostream &out, Connection & c) {
		return out << "<a: " << c.a << "; b: " << c.b << "; sim: " << c.similarity << ">";
	}

	bool operator<(const Connection & rhs) const {
		return rhs.similarity < similarity;
	}

	int a;
	int b;
	float similarity;
};

#endif
