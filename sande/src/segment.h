#ifndef SEGMENT_H_
#define SEGMENT_H_

#include <opencv2/opencv.hpp>

enum {
	COLOR_SIMILARITY   = 0x1,
	TEXTURE_SIMILARITY = 0x2,
	SIZE_SIMILARITY    = 0x4,
	BBOX_SIMILARITY    = 0x8,
};

const cv::Mat se = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,3));

#define min(a, b) (a < b ? a : b)
#define max(a, b) (a > b ? a : b)

#define HIST_SIZE 25
#define HIST_STEP int(255 / HIST_SIZE)

class Segment
{
public:
	Segment(int id, int nchannels, cv::Size size) :
		mask(cv::Mat::zeros(size, CV_8UC1)),
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
	void addPoint(cv::Vec<_Tp, n> & col, uint8_t text, cv::Point point) {
		size++;

		for (int i = 0; i < n; i++)
			color_hist[i*HIST_SIZE + col[i] / HIST_STEP]++;

		texture_hist[text / HIST_STEP]++;

		mask.at<uint8_t>(point) = 1;

		min_p = cv::Point(min(point.x, min_p.x), min(point.y, min_p.y));
		max_p = cv::Point(max(point.x, max_p.x), max(point.y, max_p.y));
	}

	inline void addNeighbour(int n) {
		neighbours.insert(n);
	}

	void normalizeHistogram() {
		cv::normalize(color_hist, color_hist, 1, 0, cv::NORM_L1);
		cv::normalize(texture_hist, texture_hist, 1, 0, cv::NORM_L1);
	}

	static double computeSimilarity(Segment & a, Segment & b, uint8_t flags)
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

		double similarity = 0.f;

		if (flags & COLOR_SIMILARITY) {
			for (uint64_t i = 0; i < a.color_hist.size(); i++)
				similarity += min(a.color_hist[i], b.color_hist[i]);
		}

		if (flags & TEXTURE_SIMILARITY) {
			for (uint64_t i = 0; i < a.texture_hist.size(); i++)
				similarity += min(a.texture_hist[i], b.texture_hist[i]);
		}

		if (flags & SIZE_SIMILARITY) {
			similarity += 1.0 - double(a.size + b.size) / a.im_size;
		}

		if (flags & BBOX_SIMILARITY) {
			cv::Point min_p(min(a.min_p.x, b.min_p.x), min(a.min_p.y, b.min_p.y));
			cv::Point max_p(max(a.max_p.x, b.max_p.x), max(a.max_p.y, b.max_p.y));
			int bbox_size = (max_p.x - min_p.x) * (max_p.y - min_p.y);
			similarity += 1.f - double(bbox_size - a.size - b.size) / a.im_size;
		}

		return similarity;
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

	std::set<int> neighbours;
	std::set<int> history;
	cv::Mat mask;
	double size;
	int im_size;
	cv::Size im_size_cv;
	std::vector<float> color_hist;
	std::vector<float> texture_hist;
	cv::Point min_p;
	cv::Point max_p;
	int id;
	int nchannels;
};

#endif
