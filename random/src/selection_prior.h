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

		return -1;
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
