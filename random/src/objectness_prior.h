#pragma once

#include "selection_prior.h"
#include "Objectness/stdafx.h"
#include "Objectness/Objectness.h"

class ObjectnessPrior : public SelectionPrior {
public:
	ObjectnessPrior() :
		obj_bbox(2, 8, 1)
	{
		obj.loadTrainedModel("/Users/hans/MasterThesis/code/cpp/random/model/ObjNessB2W8MAXBGR");
		obj_bbox.loadTrainedModel("/Users/hans/MasterThesis/code/cpp/random/model/ObjNessB2W8MAXBGR");
	}

	float scoreBBox(const cv::Mat & img) {
		cv::Mat img_small;
		if (img.size() != cv::Size(8,8)) {
			cv::resize(img, img_small, cv::Size(8,8), 0, 0, cv::INTER_CUBIC);
		} else
			img_small = img;

		ValStructVec<float, Vec4i> boxes;
		obj_bbox.getObjBndBoxes(img_small, boxes, 1);
		for (int i = 0; i < boxes.size(); i++) {
			std::cout << "value: " << boxes(i) << " rect: " << boxes[i] << std::endl;
		}

		return boxes(0);
	}

	float boxOverlap(cv::Vec4i bbox1, cv::Vec4i bbox2) {
		cv::Vec4i bi(max(bbox1[0],bbox2[0]), max(bbox1[1],bbox2[1]), min(bbox1[2],bbox2[2]), min(bbox1[3],bbox2[3]));
		int iw = bi[2] - bi[0];
		int ih = bi[3] - bi[1];
		if (iw > 0 && ih > 0) {
			float overlap = iw * ih / float((bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1]));
			return overlap;
		}

		return 0.f;
	}

	virtual SelectionPriorMap computeSelectionPrior(const cv::Mat & image, const std::vector<std::shared_ptr<Segment>> & segments) {
		SelectionPriorMap prior;


		ValStructVec<float, Vec4i> boxes;
		obj.getObjBndBoxes(image, boxes, 130);
		//std::cout << "estimated: " << 
// 		scoreBBox(image(cv::Range(boxes[0][1], boxes[0][3]), cv::Range(boxes[0][0], boxes[0][2])));
// 		std::cout << "actual: " << boxes(0) << std::endl;

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

		float sum = 0.f;
		for (const auto & s: segments) {
			float score = 0.f;
			for (int i = 0; i < boxes.size(); i++) {
				//std::cout << jaccardSimilarity(boxes[i], cv::Vec4i(s.min_p.x, s.min_p.y, s.max_p.x, s.max_p.y)) << std::endl; 
				if (boxOverlap(boxes[i], cv::Vec4i(s->min_p.x, s->min_p.y, s->max_p.x, s->max_p.y)) > 0.3)
					score += boxes(i);
			}
			sum += score;
			prior.insert({s->id, score});
		}

// 		cv::Mat likelihood = cv::Mat::zeros(image.size(), CV_32FC1);
// 		float max = boxes(0);
// 		float min = boxes(boxes.size() - 1);
// 		for (int i = 0; i < boxes.size(); i++) {
// 			float score = (boxes(i) - min) / (max - min); // normalized score
// // 			std::cout << "score: " << score << std::endl;
// 			for (int x = boxes[i][0] - 1; x < boxes[i][2]; x++) {
// 				for (int y = boxes[i][1] - 1; y < boxes[i][3]; y++) {
// 					likelihood.at<float>(y, x) += score;
// 				}
// 			}
// 		}
// 
// 		float sum = 0.f;
// 		for (const auto & s: segments) {
// 			cv::Mat mask = cv::Mat::zeros(image.size(), CV_8UC1);
// 			cv::rectangle(mask, s.min_p, s.max_p, cv::Scalar(1), CV_FILLED);
// 			float score = cv::mean(likelihood, mask)[0];
// 			sum += score;
// 			prior.insert({s.id, score});
// 		}

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
	Objectness obj_bbox;
};
