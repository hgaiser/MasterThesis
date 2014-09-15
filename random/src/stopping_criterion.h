#pragma once

#include <opencv2/opencv.hpp>

class StoppingCriterion {
public:
	virtual bool stop(const cv::Mat & image, cv::Rect roi) = 0;
};
