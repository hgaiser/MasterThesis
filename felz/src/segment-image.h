#ifndef SEGMENT_IMAGE
#define SEGMENT_IMAGE

#include "segment-graph.h"
#include <map>

#define square(x) ((x)*(x))
//#define COLORED_OUTPUT

// random color
cv::Vec3b random_rgb(){ 
	cv::Vec3b c;
	double r;
	c[0] = (uchar)random();
	c[1] = (uchar)random();
	c[2] = (uchar)random();
	return c;
}

/// Difference between two pixels p1, p2 in image
static inline float diff(const cv::Mat & image, int p1, int p2) {
	if (image.channels() == 3) {
		cv::Vec3f a = image.at<cv::Vec3f>(p1);
		cv::Vec3f b = image.at<cv::Vec3f>(p2);
		return sqrt(square(a[0] - b[0]) + square(a[1] - b[1]) + square(a[2] - b[2]));
	} else
		return fabs(image.at<float>(p1) - image.at<float>(p2));
}

/*
 * Segment an image
 *
 * Returns a color image representing the segmentation.
 *
 * im: image to segment.
 * sigma: to smooth the image.
 * c: constant for treshold function.
 * min_size: minimum component size (enforced by post-processing stage).
 */
cv::Mat segment_image(cv::Mat image, float sigma, float c, int min_size) {
	image.convertTo(image, CV_MAKETYPE(CV_32F, image.channels()));
	cv::GaussianBlur(image, image, cv::Size(5, 5), sigma);
	//image = smooth(image, sigma);

	// build graph
	std::vector<Edge> edges;
	edges.resize(image.total() * 4);
	int index = 0;
	for (int i = 0; i < image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {
			int v1 = i * image.cols + j;

			if (j < image.cols - 1) {
				int v2 = i * image.cols + (j+1);
				Edge & e = edges[index++];
				e.setFirst(v1);
				e.setSecond(v2);
				e.setWeight(diff(image, v1, v2));
			}
			if (i < image.rows - 1) {
				int v2 = (i+1) * image.cols + j;
				Edge & e = edges[index++];
				e.setFirst(v1);
				e.setSecond(v2);
				e.setWeight(diff(image, v1, v2));
			}
			if (i < image.rows - 1 && j < image.cols - 1) {
				int v2 = (i+1) * image.cols + (j+1);
				Edge & e = edges[index++];
				e.setFirst(v1);
				e.setSecond(v2);
				e.setWeight(diff(image, v1, v2));
			}
			if (i > 0 && j < image.cols - 1) {
				int v2 = (i-1) * image.cols + (j+1);
				Edge & e = edges[index++];
				e.setFirst(v1);
				e.setSecond(v2);
				e.setWeight(diff(image, v1, v2));
			}
		}
	}

	// segment
	std::vector<Cluster> clusters = segment_graph(image.total(), index, edges, c);

	// post process small components
	if (min_size) {
		for (int i = 0; i < index; i++) {
			int a = find(clusters, edges[i].first());
			int b = find(clusters, edges[i].second());
			if ((a != b) && ((clusters[a].size() < min_size) || (clusters[b].size() < min_size))) {
				join(clusters, a, b);
			}
		}
	}

#ifdef COLORED_OUTPUT
	cv::Mat output(image.size(), CV_8UC3);

	// pick random colors for each component
	std::vector<cv::Vec3b> colors;
	colors.resize(image.total());
	for (int i = 0; i < image.total(); i++)
		colors[i] = random_rgb();
  
	for (int y = 0; y < image.rows; y++) {
		for (int x = 0; x < image.cols; x++) {
			int comp = find(clusters, y * image.cols + x);
			output.at<cv::Vec3b>(y, x) = colors[comp];
		}
	}
#else
	cv::Mat output(image.size(), CV_16UC1);
	std::map<int, uint16_t> cluster_ids;
	uint16_t max_id = 1;

	for (int y = 0; y < image.rows; y++) {
		for (int x = 0; x < image.cols; x++) {
			int comp = find(clusters, y * image.cols + x);
			if (cluster_ids.find(comp) == cluster_ids.end())
				cluster_ids[comp] = max_id++;
			output.at<uint16_t>(y, x) = cluster_ids[comp];
		}
	}
#endif

	return output;
}

#endif
