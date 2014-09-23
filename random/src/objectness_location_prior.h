#pragma once

#include "selection_prior.h"
#include "Objectness/stdafx.h"
#include "Objectness/Objectness.h"

class ObjectnessLocationPrior : public SelectionPrior {
public:
	ObjectnessLocationPrior() {
		obj.loadTrainedModel("/Users/hans/MasterThesis/code/cpp/random/model/ObjNessB2W8MAXBGR");
	}

	virtual SelectionPriorMap computeSelectionPrior(const cv::Mat & image, const std::vector<std::unique_ptr<Segment>> & segments) {
		SelectionPriorMap prior;

		cv::Mat likelihood = cv::Mat::zeros(image.size(), CV_32FC1);

		ValStructVec<float, Vec4i> boxes;
		obj.getObjBndBoxes(image, boxes, 130);

// 		cv::Mat test = cv::imread("../../../images/horse.jpg");
// 		cv::Mat test; image.copyTo(test);
// 		for (int i = 0; i < boxes.size(); i++) {
// 			cv::Mat draw;
// 			test.copyTo(draw);
// 			cv::rectangle(draw, cv::Point(boxes[i][0], boxes[i][1]), cv::Point(boxes[i][2], boxes[i][3]), cv::Scalar(255, 255, 255));
// 			std::cout << boxes(i) << std::endl;
// 			cv::namedWindow("Objectness", 0);
// 			cv::imshow("Objectness", draw);
// 			cv::waitKey();
// 		}

		float max = boxes(0);
		float min = boxes(boxes.size() - 1);
		for (int i = 0; i < boxes.size(); i++) {
			float score = (boxes(i) - min) / (max - min); // normalized score
// 			std::cout << "score: " << score << std::endl;
			for (int x = boxes[i][0] - 1; x < boxes[i][2]; x++) {
				for (int y = boxes[i][1] - 1; y < boxes[i][3]; y++) {
					likelihood.at<float>(y, x) += score;
				}
			}
		}

		//uint32_t radius = (image.cols*image.cols + image.rows*image.rows) / 4;
		//cv::Point center = cv::Point(image.cols / 2, image.rows / 2);

		float sum = 0.f;
		for (const auto & s: segments) {
			cv::Mat mask = cv::Mat::zeros(image.size(), CV_8UC1);
			cv::rectangle(mask, s->min_p, s->max_p, cv::Scalar(1), CV_FILLED);
			float score = cv::mean(likelihood, mask)[0];

			//cv::Point diff = (s.min_p + s.max_p) * 0.5 - center;
			//score *= exp(-diff.dot(diff) / float(radius) * deviation);

			sum += score;
			prior.insert({s->id, score});
		}

		for (auto & p: prior) {
			p.second /= sum;
		}

// 		cv::normalize(likelihood, likelihood, 0.f, 1.f, cv::NORM_MINMAX);
// 		cv::namedWindow("Likelihood", cv::WINDOW_AUTOSIZE);
// 		cv::namedWindow("Image", cv::WINDOW_AUTOSIZE);
// 		cv::imshow("Likelihood", likelihood);
// 		cv::imshow("Image", image);
// 		cv::waitKey();

		return prior;
	}

protected:
	Objectness obj;
};
