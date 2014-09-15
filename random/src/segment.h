#pragma once

#include <opencv2/opencv.hpp>

enum {
	COLOR_SIMILARITY   = 0x1,
	TEXTURE_SIMILARITY = 0x2,
	SIZE_SIMILARITY    = 0x4,
	BBOX_SIMILARITY    = 0x8,
};

#define HIST_SIZE 25
#define HIST_STEP int(255 / HIST_SIZE)

class Segment
{
public:
	Segment() {}
	Segment(int id, int nchannels, cv::Size size) :
#ifdef DEBUG
		mask(cv::Mat::zeros(size, CV_8UC1)),
#endif
		size(0),
		im_size(size.area()),
		im_size_cv(size),
		min_p(size.width, size.height),
		max_p(0, 0),
		id(id),
		nchannels(nchannels)
	{
		color_hist.assign(nchannels * HIST_SIZE, 0.f);
		texture_hist.assign(HIST_SIZE, 0.f);
		history.insert(id);
	}

	template<typename _Tp, int n>
	void addPoint(cv::Vec<_Tp, n> col, uint8_t text, cv::Point point) {
		size++;

		for (int i = 0; i < n; i++)
			color_hist[i*HIST_SIZE + col[i] / HIST_STEP]++;

		texture_hist[text / HIST_STEP]++;

#ifdef DEBUG
		mask.at<uint8_t>(point) = 1;
#endif

		min_p = cv::Point(std::min(point.x, min_p.x), std::min(point.y, min_p.y));
		max_p = cv::Point(std::max(point.x, max_p.x), std::max(point.y, max_p.y));
	}

	inline void addNeighbour(int n) {
		neighbours.insert(n);
	}

	void normalizeHistogram() {
		cv::normalize(color_hist, color_hist, 1, 0, cv::NORM_L1);
		cv::normalize(texture_hist, texture_hist, 1, 0, cv::NORM_L1);
	}

	static float computeSimilarity(const Segment & a, const Segment & b, uint8_t flags)
	{
		if (a.im_size != b.im_size) {
			std::cerr << "Error! Segments are from different images?" << std::endl;
			return 0.f;
		}
		if (a.size == 0 || b.size == 0) {
			std::cout << "Warning! Invalid segment." << std::endl;
			return 0.f;
		}
		if (a.color_hist.size() != b.color_hist.size() || a.texture_hist.size() != b.texture_hist.size()) {
			std::cout << "Error! Segments histograms don't match!" << std::endl;
			return 0.f;
		}

		float similarity = 0.f;

		if (flags & COLOR_SIMILARITY) {
			for (uint64_t i = 0; i < a.color_hist.size(); i++)
				similarity += std::min(a.color_hist[i], b.color_hist[i]);
		}

		if (flags & TEXTURE_SIMILARITY) {
			for (uint64_t i = 0; i < a.texture_hist.size(); i++)
				similarity += std::min(a.texture_hist[i], b.texture_hist[i]);
		}

		if (flags & SIZE_SIMILARITY) {
			similarity += 1.f - float(a.size + b.size) / a.im_size;
		}

		if (flags & BBOX_SIMILARITY) {
			cv::Point min_p(std::min(a.min_p.x, b.min_p.x), std::min(a.min_p.y, b.min_p.y));
			cv::Point max_p(std::max(a.max_p.x, b.max_p.x), std::max(a.max_p.y, b.max_p.y));
			int bbox_size = (max_p.x - min_p.x) * (max_p.y - min_p.y);
			similarity += 1.f - float(bbox_size - a.size - b.size) / a.im_size;
		}

		return similarity;
	}

	static Segment merge(const Segment & a, const Segment & b) {
		Segment s(-1, a.nchannels, a.im_size_cv);
		s.size = a.size + b.size;
#ifdef DEBUG
		cv::bitwise_or(a.mask, b.mask, s.mask);
#endif

		if (a.color_hist.size() != b.color_hist.size() || a.texture_hist.size() != b.texture_hist.size())
			return s;

		for (uint64_t i = 0; i < a.color_hist.size(); i++) {
			s.color_hist[i] = (a.color_hist[i] * a.size + b.size * b.color_hist[i]) / s.size;
		}

		for (uint64_t i = 0; i < a.texture_hist.size(); i++)
			s.texture_hist[i] = (a.size * a.texture_hist[i] + b.size * b.texture_hist[i]) / s.size;
		//s.texture_hist = (a.size * a.texture_hist + b.size * b.texture_hist) / s.size;

		s.min_p = cv::Point(std::min(a.min_p.x, b.min_p.x), std::min(a.min_p.y, b.min_p.y));
		s.max_p = cv::Point(std::max(a.max_p.x, b.max_p.x), std::max(a.max_p.y, b.max_p.y));

		s.history.insert(a.history.begin(), a.history.end());
		s.history.insert(b.history.begin(), b.history.end());

		s.neighbours.insert(a.neighbours.begin(), a.neighbours.end());
		s.neighbours.insert(b.neighbours.begin(), b.neighbours.end());

		for (const auto & h: s.history)
			s.neighbours.erase(h);

		return s;
	}

	friend std::ostream& operator<<(std::ostream &out, Segment & s) {
		out << "<id: " << s.id << "; size: " << s.size << "; im_size: " << s.im_size << "; bbox: " << s.bbox() << "; neighbours: [";
		for (auto n: s.neighbours)
			out << n << ", ";
		return out << "]>";
	}

	inline cv::Rect bbox() {
		return cv::Rect(min_p, max_p);
	}

	inline bool empty() const {
		return size == 0;
	}

	std::set<int> neighbours;
	std::set<int> history;
#ifdef DEBUG
	cv::Mat mask;
#endif
	int size;
	int im_size;
	cv::Size im_size_cv;
	std::vector<float> color_hist;
	std::vector<float> texture_hist;
	cv::Point min_p;
	cv::Point max_p;
	int id;
	int nchannels;
};

