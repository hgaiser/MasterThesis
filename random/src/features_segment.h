#pragma once

#include <vector>
#include <opencv2/opencv.hpp>
#include "segment.h"

enum {
	COLOR_SIMILARITY   = 0x1,
	TEXTURE_SIMILARITY = 0x2,
	SIZE_SIMILARITY    = 0x4,
	BBOX_SIMILARITY    = 0x8,
};

#define HIST_SIZE 25
#define HIST_STEP int(255 / HIST_SIZE)

class FeaturesSegment: public Segment
{
public:
	FeaturesSegment(int id, int nchannels, cv::Size size, uint8_t f, const cv::Mat & w) :
		Segment(id, size),
		nchannels(nchannels),
		flags(f),
		weights(w)
	{
		color_hist.assign(nchannels * HIST_SIZE, 0.f);
		texture_hist.assign(HIST_SIZE, 0.f);
	}

	virtual void addPoint(const cv::Mat & image, const cv::Mat & texture, cv::Point point) override {
		Segment::addPoint(image, texture, point);
		size++;
		if (nchannels == 1) {
			cv::Vec<uint8_t, 1> color = image.at<cv::Vec<uint8_t, 1>>(point);
			for (int i = 0; i < nchannels; i++)
				color_hist[i*HIST_SIZE + color[i] / HIST_STEP]++;
		} else if (nchannels == 3) {
			cv::Vec<uint8_t, 3> color = image.at<cv::Vec<uint8_t, 3>>(point);
			for (int i = 0; i < nchannels; i++)
				color_hist[i*HIST_SIZE + color[i] / HIST_STEP]++;
		}
		uint8_t text = texture.at<uint8_t>(point);
		texture_hist[text / HIST_STEP]++;
	}

	virtual void finalizeSetup() override {
		cv::normalize(color_hist, color_hist, 1, 0, cv::NORM_L1);
		cv::normalize(texture_hist, texture_hist, 1, 0, cv::NORM_L1);
	}

	virtual float computeSimilarity(const Segment * b_) override {
		const FeaturesSegment * b = dynamic_cast<const FeaturesSegment *>(b_);
		if (weights.type() != CV_32FC1) {
			std::cerr << "Error! Weights are not one channel float " << weights << std::endl;
			return 0.f;
		}
		if (weights.total() != 5) {
			std::cerr << "Error! Not enough weights: " << weights << std::endl;
			return 0.f;
		}
		if (im_size != b->im_size) {
			std::cerr << "Error! Segments are from different images?" << std::endl;
			return 0.f;
		}
		if (size == 0 || b->size == 0) {
			std::cout << "Warning! Invalid segment." << std::endl;
			return 0.f;
		}
		if (color_hist.size() != b->color_hist.size() || texture_hist.size() != b->texture_hist.size()) {
			std::cout << "Error! Segments histograms don't match!" << std::endl;
			return 0.f;
		}

		float color_similarity = 0.f;
		if (flags & COLOR_SIMILARITY) {
			for (uint64_t i = 0; i < color_hist.size(); i++)
				color_similarity += weights.at<float>(0) * std::min(color_hist[i], b->color_hist[i]);
		}

		float texture_similarity = 0.f;
		if (flags & TEXTURE_SIMILARITY) {
			for (uint64_t i = 0; i < texture_hist.size(); i++)
				texture_similarity += weights.at<float>(1) * std::min(texture_hist[i], b->texture_hist[i]);
		}

		float size_similarity = 0.f;
		if (flags & SIZE_SIMILARITY) {
			size_similarity += weights.at<float>(2) * (1.f - float(size + b->size) / im_size);
		}

		float bbox_similarity = 0.f;
		if (flags & BBOX_SIMILARITY) {
			cv::Point combined_min(std::min(min_p.x, b->min_p.x), std::min(min_p.y, b->min_p.y));
			cv::Point combined_max(std::max(max_p.x, b->max_p.x), std::max(max_p.y, b->max_p.y));
			int bbox_size = (combined_max.x - combined_min.x) * (combined_max.y - combined_min.y);
			bbox_similarity += weights.at<float>(3) * (1.f - float(bbox_size - size - b->size) / im_size);
		}

		return color_similarity + texture_similarity + size_similarity + bbox_similarity + weights.at<float>(4);
	}

	std::unique_ptr<Segment> merge(const Segment * b_) override {
		const FeaturesSegment * b = dynamic_cast<const FeaturesSegment *>(b_);
		std::unique_ptr<Segment> s_(new FeaturesSegment(-1, nchannels, im_size_cv, flags, weights));
		FeaturesSegment * s = dynamic_cast<FeaturesSegment *>(s_.get());
		s->size = size + b->size;
#ifdef DEBUG
		cv::bitwise_or(mask, b->mask, s->mask);
#endif

		s->min_p = cv::Point(std::min(min_p.x, b->min_p.x), std::min(min_p.y, b->min_p.y));
		s->max_p = cv::Point(std::max(max_p.x, b->max_p.x), std::max(max_p.y, b->max_p.y));

		if (color_hist.size() != b->color_hist.size() || texture_hist.size() != b->texture_hist.size())
			return s_;

		for (uint64_t i = 0; i < color_hist.size(); i++) {
			s->color_hist[i] = (color_hist[i] * size + b->size * b->color_hist[i]) / s->size;
		}

		for (uint64_t i = 0; i < texture_hist.size(); i++)
			s->texture_hist[i] = (size * texture_hist[i] + b->size * b->texture_hist[i]) / s->size;
		//s->texture_hist = (size * texture_hist + b->size * b->texture_hist) / s->size;


		s->history.insert(history.begin(), history.end());
		s->history.insert(b->history.begin(), b->history.end());

		s->neighbours.insert(neighbours.begin(), neighbours.end());
		s->neighbours.insert(b->neighbours.begin(), b->neighbours.end());

		for (const auto & h: s->history)
			s->neighbours.erase(h);

		return s_;
	}

	std::ostream & output(std::ostream & out) const {
		out << "<id: " << id << "; size: " << size << "; im_size: " << im_size << "; bbox: " << bbox() << "; neighbours: [";
		for (auto n: neighbours)
			out << n << ", ";
		return out << "]>";
	}

	inline cv::Rect bbox() const {
		return cv::Rect(min_p, max_p);
	}

	std::vector<float> color_hist;
	std::vector<float> texture_hist;
	int nchannels;
	uint8_t flags;
	const cv::Mat & weights;
};
