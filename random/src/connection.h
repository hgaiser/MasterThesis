#pragma once

#include "segment.h"

class Connection
{
public:
	Connection(const Segment & a_, const Segment & b_, uint8_t flags, const cv::Mat & weights) :
		a(a_.id), b(b_.id), similarity(exp(Segment::computeSimilarity(a_, b_, flags, weights)))
	{
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

