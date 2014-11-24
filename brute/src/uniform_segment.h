#pragma once

#include "segment.h"

class UniformSegment : public Segment {
public:
	UniformSegment(int id, cv::Size size) :
		Segment(id, size)
	{
	}

	virtual float computeSimilarity(const Segment * b_) override {
		return 1.f;
	}

	virtual std::shared_ptr<Segment> merge(const Segment * b_) override {
		const UniformSegment * b = dynamic_cast<const UniformSegment *>(b_);
		std::shared_ptr<Segment> s_ = std::make_shared<UniformSegment>(-1, im_size_cv);
		UniformSegment * s = dynamic_cast<UniformSegment *>(s_.get());
		s->size = size + b->size;
#ifdef DEBUG
		cv::bitwise_or(mask, b->mask, s->mask);
#endif

		s->min_p = cv::Point(std::min(min_p.x, b->min_p.x), std::min(min_p.y, b->min_p.y));
		s->max_p = cv::Point(std::max(max_p.x, b->max_p.x), std::max(max_p.y, b->max_p.y));

		s->history.insert(history.begin(), history.end());
		s->history.insert(b->history.begin(), b->history.end());

		s->neighbours.insert(neighbours.begin(), neighbours.end());
		s->neighbours.insert(b->neighbours.begin(), b->neighbours.end());

		for (const auto & h: s->history)
			s->neighbours.erase(h);

		return s_;
	}

	virtual std::ostream & output(std::ostream & out) const override {
		out << "<id: " << id << "; size: " << size << "; im_size: " << im_size << "; neighbours: [";
		for (auto n: neighbours)
			out << n << ", ";
		return out << "]>";
	}
};
