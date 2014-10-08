#pragma once

#include <opencv2/opencv.hpp>

class Segment
{
public:
	Segment(int id, cv::Size size) :
#ifdef DEBUG
		mask(cv::Mat::zeros(size, CV_8UC1)),
#endif
		id(id),
		size(0),
		min_p(size.width, size.height),
		max_p(0, 0),
		im_size(size.area()),
		im_size_cv(size)
	{
		history.insert(id);
	}

	virtual void addPoint(const cv::Mat & image, const cv::Mat & texture, cv::Point point) {
		size++;
#ifdef DEBUG
		mask.at<uint8_t>(point) = 1;
#endif

		min_p = cv::Point(std::min(point.x, min_p.x), std::min(point.y, min_p.y));
		max_p = cv::Point(std::max(point.x, max_p.x), std::max(point.y, max_p.y));
	}

	inline void addNeighbour(int n) {
		neighbours.insert(n);
	}

	virtual float computeSimilarity(const Segment * b_) = 0;

	virtual std::shared_ptr<Segment> merge(const Segment * b) = 0;

	virtual void finalizeSetup() {}

	virtual std::ostream & output(std::ostream &) const = 0;

	inline bool empty() const {
		return size == 0;
	}

	inline cv::Rect bbox() const {
		return cv::Rect(min_p, max_p);
	}

	int size;
	std::set<int> neighbours;
	std::set<int> history;
#ifdef DEBUG
	cv::Mat mask;
#endif
	int id;
	cv::Point min_p;
	cv::Point max_p;
	int im_size;
	cv::Size im_size_cv;
};

std::ostream & operator<<(std::ostream & os, const Segment & b) {
    return b.output(os);
}

