#ifndef SEGMENT_IMAGE
#define SEGMENT_IMAGE

#include "segment-graph.h"

#define square(x) ((x)*(x))

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
static inline float diff(cv::Mat image, int p1, int p2) {
	//cv::Vec3b d = image.at<cv::Vec3b>(p1) - image.at<cv::Vec3b>(p2);
	cv::Vec3f a = image.at<cv::Vec3f>(p1);
	cv::Vec3f b = image.at<cv::Vec3f>(p2);
	return sqrt(square(a[0] - b[0]) + square(a[1] - b[1]) + square(a[2] - b[2]));
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
 * num_ccs: number of connected components in the segmentation.
 */
cv::Mat segment_image(cv::Mat image, float sigma, float c, int min_size, int & num_ccs) {
	image.convertTo(image, CV_32FC3);
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

	num_ccs = image.total();

	// segment
	std::vector<Cluster> clusters = segment_graph(image.total(), index, edges, c, num_ccs);

	// post process small components
	if (min_size) {
		for (int i = 0; i < index; i++) {
			int a = find(clusters, edges[i].first());
			int b = find(clusters, edges[i].second());
			if ((a != b) && ((clusters[a].size() < min_size) || (clusters[b].size() < min_size))) {
				join(clusters, a, b);
				num_ccs--;
			}
		}
	}

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

	return output;
}

#endif
