#pragma once

#include "selection_prior.h"

class UniformPrior : public SelectionPrior {
public:
	virtual SelectionPriorMap computeSelectionPrior(const cv::Mat & image, const std::vector<std::unique_ptr<Segment>> & segments) {
		SelectionPriorMap prior;

		int sum = 0;
		for (const auto & s: segments) {
			if (s->empty())
				continue;

			sum++;
			prior.insert({s->id, 1});
		}

		for (auto & p: prior) {
			p.second /= sum;
		}

		return prior;
	}
};
