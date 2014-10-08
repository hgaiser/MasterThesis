#pragma once

#include <vector>
#include <opencv2/opencv.hpp>

bool startCuda();

std::vector<cv::Mat> computeSegmentationLevels_(const cv::Mat & image);

void stopCuda();
