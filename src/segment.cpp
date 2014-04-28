#include <iostream>
#include <time.h>
#include <map>
#include <ctime>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "edge.h"

bool sort(Edge a, Edge b) { return a.weight() < b.weight(); }

// TODO: compare with struct in a vector?
// cluster data:
// [0] = id
// [1] = threshold
// [2] = size of cluster
// [3] = rank (number of times merged with other cluster)

/// Join two clusters
void join(cv::Vec4f & c1, cv::Vec4f & c2, int k, float weight) {
	if (c1[3] > c2[3]) {
		c2[0] = c1[0];
		c1[2] += c2[2];
		c1[1] = weight + k / c1[2];
	} else {
		c1[0] = c2[0];
		c2[2] += c1[2];
		if (c1[3] == c2[3])
			c2[3]++;
		c2[1] = weight + k / c2[2];
	}
}

/// Difference between two pixels p1, p2 in image
float diff(cv::Mat image, int p1, int p2) {
	cv::Vec3b d = image.at<cv::Vec3b>(p1) - image.at<cv::Vec3b>(p2);
	return sqrt(d[0]*d[0] + d[1]*d[1] + d[2]*d[2]);
}


/// Traverses the path in clusters to find the end cluster for id
cv::Vec4f& find(cv::Mat & clusters, int id) {
	cv::Vec4f & c = clusters.at<cv::Vec4f>(id);
	cv::Vec4f & c_ = clusters.at<cv::Vec4f>(c[0]);
	while (c[0] != c_[0]) {
		c = c_;
		c_ = clusters.at<cv::Vec4f>(c[0]);
	}
	clusters.at<cv::Vec4f>(id)[0] = c[0];
	return c;
}

/// Main segment method
cv::Mat segment(cv::Mat image, double sigma, int k, int min_size) {
	cv::GaussianBlur(image, image, cv::Size(3, 3), sigma);

	//cv::Mat clusters(image.size(), CV_32SC1);
	cv::Mat clusters = cv::Mat::zeros(image.size(), CV_32FC4);
	//cv::Mat sizes(image.size(), CV_32SC1);

	std::vector<Edge> edges;
	edges.resize(image.rows*image.cols*4);

	// add edges
	int index = 0;
	for (int i = 1; i < image.rows - 1; i++) {
		for (int j = 0; j < image.cols - 1; j++) {
			int v1_ = i * image.cols + j;
			cv::Vec4f & c = clusters.at<cv::Vec4f>(i, j);
			c[0] = v1_;
			c[1] = k;

			// right
			//if (j < image.cols - 1) {
				int v2_ = v1_ + 1;
				//int v2_ = i * image.cols + (j+1);
				Edge & e = edges[index++];
				e.setFirst(v1_);
				e.setSecond(v2_);
				e.setWeight(diff(image, v1_, v2_));
			//}
			// down
			//if (i < image.rows - 1) {
				v2_ = v1_ + image.cols;
				//int v2_ = (i+1) * image.cols + j;
				e = edges[index++];
				e.setFirst(v1_);
				e.setSecond(v2_);
				e.setWeight(diff(image, v1_, v2_));
			//}
			// down right
			//if (i < image.rows - 1 && j < image.cols - 1) {
				v2_ = v1_ + image.cols + 1;
				//int v2_ = (i+1) * image.cols + (j+1);
				e = edges[index++];
				e.setFirst(v1_);
				e.setSecond(v2_);
				e.setWeight(diff(image, v1_, v2_));
			//}
			// up right
			//if (i > 0 && j < image.cols - 1) {
				v2_ = v1_ - image.cols + 1;
				//int v2_ = (i-1) * image.cols + (j+1);
				e = edges[index++];
				e.setFirst(v1_);
				e.setSecond(v2_);
				e.setWeight(diff(image, v1_, v2_));
			//}
		}
	}

	// add cluster data for skipped borders
	// last column
	for (int i = 0; i < image.rows; i++) {
		cv::Vec4f & c = clusters.at<cv::Vec4f>(i, image.cols-1);
		c[0] = i * image.cols + image.cols - 1;
		c[1] = k;
	}
	// first row
	for (int j = 0; j < image.cols; j++) {
		cv::Vec4f & c = clusters.at<cv::Vec4f>(0, j);
		c[0] = j;
		c[1] = k;
	}
	// last row
	for (int j = 0; j < image.cols; j++) {
		cv::Vec4f & c = clusters.at<cv::Vec4f>(image.rows-1, j);
		c[0] = (image.rows-1) * image.cols + j;
		c[1] = k;
	}

	//std::clock_t begin = std::clock();
	std::sort(edges.begin(), edges.begin() + index, sort);
	//std::clock_t end = std::clock();
	//double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
	//std::cout << "Times passed in seconds: " << elapsed_secs << std::endl;

	// merge the clusters
	for (auto e: edges) {
		cv::Vec4f & c1 = find(clusters, e.first());
		cv::Vec4f & c2 = find(clusters, e.second());
		if (c1[0] != c2[0] && 
			e.weight() <= c1[1] &&
			e.weight() <= c2[1]) {
			join(c1, c2, k, e.weight());
		}
	}

	// choose random colors
	cv::Mat colors(image.size(), CV_8UC3);
	for (int i = 0; i < colors.total(); i++) {
		colors.at<cv::Vec3b>(i) = {uint8_t(rand() % 256), uint8_t(rand() % 256), uint8_t(rand() % 256)};
	}

	//std::map<uint32_t, cv::Vec3b> colormap;

	// color the segmentation
	cv::Mat segmentation = cv::Mat::zeros(image.size(), CV_8UC3);
	for (int i = 1; i < image.rows - 1; i++) {
		for (int j = 0; j < image.cols - 1; j++) {
			int index = i * image.cols + j;
			cv::Vec4f & c = find(clusters, index);
			segmentation.at<cv::Vec3b>(index) = colors.at<cv::Vec3b>(c[0]);
		}
	}
	/*for (auto v: vertices) {
		auto color = colormap.find(v.cluster());
		if (color == colormap.end()) {
			colormap[v.cluster()] = cv::Vec3b(rand() % 256, rand() % 256, rand() % 256);
		}
		segmentation.at<cv::Vec3b>(v.row(), v.col()) = colormap[v.cluster()];
	}*/

	return segmentation;
}

int main(int argc, char * argv[]) {
	if (argc != 2) {
		std::cout << "Provide image as first argument." << std::endl;
		return 0;
	}

	cv::namedWindow("Image", cv::WINDOW_NORMAL);
	cv::Mat image = cv::imread(argv[1]);

	cv::Mat segmentation;
	std::clock_t begin = std::clock();
	for (int i = 0; i < 10; i++)
		segmentation = segment(image, 0.8, 300, 20);
	std::clock_t end = std::clock();
	double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC / 10;
	std::cout << "Times passed in seconds: " << elapsed_secs << std::endl;

	cv::imshow("Image", segmentation);
	cv::waitKey();
	return 0;
}
