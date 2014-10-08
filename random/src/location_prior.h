#pragma once

#include "selection_prior.h"

const float deviation = 4.f;

class LocationPrior : public SelectionPrior {
public:
	virtual SelectionPriorMap computeSelectionPrior(const cv::Mat & image, const std::vector<std::shared_ptr<Segment>> & segments) {
		SelectionPriorMap prior;

		uint32_t radius = (image.cols*image.cols + image.rows*image.rows) / 4;
		cv::Point center = cv::Point(image.cols / 2, image.rows / 2);
		float sum = 0.f;

		for (const auto & s: segments) {
			if (s->empty())
				continue;

			cv::Point diff = (s->min_p + s->max_p) * 0.5 - center;
			float likelihood = exp(-diff.dot(diff) / float(radius) * deviation);
			sum += likelihood;
			prior.insert({s->id, likelihood});
// 			std::cout << "size: " << prior.size() << " id: " << s->id << std::endl;
		}

		for (auto & p: prior) {
			p.second /= sum;
// 			std::cout << "likelihood: " << p.second << std::endl;
		}

		return prior;

/*		std::uniform_real_distribution<float> dis(0.f, sum);
		float rnd = dis(gen);
		sum = 0.f;
		for (auto p: prior) {
			sum += p.second;
			if (sum >= rnd) {
				return p.first;
			}
		}

		return 0;*/
	}
};
