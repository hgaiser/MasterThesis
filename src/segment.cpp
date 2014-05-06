#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include "segment-image.h"
#include <ctime>

int main(int argc, char **argv) {
	if (argc != 6) {
		std::cerr << "usage: " << argv[0] << " sigma k min input output" << std::endl;
		return 1;
	}
  
	float sigma = atof(argv[1]);
	float k = atof(argv[2]);
	int min_size = atoi(argv[3]);
	
	cv::Mat input = cv::imread(argv[4]);
	int num_ccs; 

	std::cout << "Processing..." << std::endl;

	std::clock_t begin = std::clock();
	cv::Mat seg = segment_image(input, sigma, k, min_size, num_ccs); 
	std::clock_t end = std::clock();
	double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
	cv::imwrite(argv[5], seg);
	std::cout << "Times passed in seconds: " << elapsed_secs << std::endl;

	std::cout << "Got " << num_ccs << " components." << std::endl;
	return 0;
}

