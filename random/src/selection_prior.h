#pragma once

#include <random>
#include <unordered_map>
#include "segment.h"

class SelectionPriorMap : public std::unordered_map<uint32_t, float> {
public:
	SelectionPriorMap() : std::unordered_map<uint32_t, float>() {}

	uint32_t poll() {
		float rnd = dis(gen);
		float sum = 0.f;
		for (auto p: *this) {
			sum += p.second;
			if (sum >= rnd) {
				return p.first;
			}
		}

		return size() - 1;
	}

	void visualize(const cv::Mat & seg) {
		float max = 0.f;
		for (auto & p: *this) {
			if (p.second > max) {
				max = p.second;
			}
		}

		cv::Mat visualization = cv::Mat::zeros(seg.size(), CV_8UC1);
		for (auto & p: *this) {
			visualization += (seg == p.first) * (p.second / max);
		}

		cv::namedWindow("SelectionPrior", cv::WINDOW_AUTOSIZE);
		cv::imshow("SelectionPrior", visualization);
		cv::waitKey();
	}

protected:
	static std::random_device rd;
	static std::mt19937 gen;
	static std::uniform_real_distribution<float> dis;
};

std::random_device SelectionPriorMap::rd;
std::mt19937 SelectionPriorMap::gen(rd());
std::uniform_real_distribution<float> SelectionPriorMap::dis(0.f, 1.f);

class SelectionPrior {
public:
	virtual SelectionPriorMap computeSelectionPrior(const cv::Mat & image, const std::vector<Segment> & segments) = 0;
};
