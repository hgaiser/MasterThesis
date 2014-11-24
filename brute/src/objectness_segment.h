#pragma once

#include <vector>
#include <opencv2/opencv.hpp>
#include "segment.h"

class ObjectnessSegment : public Segment {
public:
	ObjectnessSegment(int id, cv::Size size, const cv::Mat & likelihood_) :
		Segment(id, size),
		objectness(0.f),
		likelihood(likelihood_)
	{
	}

	virtual void addPoint(const cv::Mat & image, const cv::Mat & texture, cv::Point point) override {
		Segment::addPoint(image, texture, point);
		objectness += likelihood.at<float>(point);
	}

	virtual void finalizeSetup() {
		objectness /= size;
	}

	virtual float computeSimilarity(const Segment * b_) override {
		const ObjectnessSegment * b = dynamic_cast<const ObjectnessSegment *>(b_);
		return b->objectness - objectness;
	}

	virtual std::shared_ptr<Segment> merge(const Segment * b_) override {
		const ObjectnessSegment * b = dynamic_cast<const ObjectnessSegment *>(b_);
		std::shared_ptr<Segment> s_ = std::make_shared<ObjectnessSegment>(-1, im_size_cv, likelihood);
		ObjectnessSegment * s = dynamic_cast<ObjectnessSegment *>(s_.get());
		s->size = size + b->size;
#ifdef DEBUG
		cv::bitwise_or(mask, b->mask, s->mask);
#endif

		s->min_p = cv::Point(std::min(min_p.x, b->min_p.x), std::min(min_p.y, b->min_p.y));
		s->max_p = cv::Point(std::max(max_p.x, b->max_p.x), std::max(max_p.y, b->max_p.y));

		s->objectness = (size * objectness + b->size * b->objectness) / (size + b->size);

		s->history.insert(history.begin(), history.end());
		s->history.insert(b->history.begin(), b->history.end());

		s->neighbours.insert(neighbours.begin(), neighbours.end());
		s->neighbours.insert(b->neighbours.begin(), b->neighbours.end());

		for (const auto & h: s->history)
			s->neighbours.erase(h);

		return s_;
	}

	virtual std::ostream & output(std::ostream & out) const override {
		out << "<id: " << id << "; size: " << size << "; im_size: " << im_size << "; objectness: " << objectness << "; neighbours: [";
		for (auto n: neighbours)
			out << n << ", ";
		return out << "]>";
	}

	float objectness;

protected:
	const cv::Mat & likelihood;
};
