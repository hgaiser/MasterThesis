#pragma once

#include "stopping_criterion.h"

class RandomStoppingCriterion : public StoppingCriterion {
public:
	RandomStoppingCriterion() : threshold(0.95f) {//threshold(dis(gen)) {
	}

	bool stop(const cv::Mat & image, cv::Rect roi) {
		return (roi.height == image.rows - 1 && roi.width == image.cols - 1) || dis(gen) > threshold;
	}

protected:
	float threshold;

	static std::random_device rd;
	static std::mt19937 gen;
	static std::uniform_real_distribution<float> dis;
};

std::random_device RandomStoppingCriterion::rd;
std::mt19937 RandomStoppingCriterion::gen(rd());
std::uniform_real_distribution<float> RandomStoppingCriterion::dis(0.f, 1.f);
