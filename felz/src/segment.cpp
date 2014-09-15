#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include "segment-image.h"
#include <ctime>
//#include "CVBoostConverter.hpp"
#include "conversion.h"
#include <boost/python.hpp>

using namespace boost::python;

PyObject * segment(PyObject * image_, float sigma, int k, int min_size) {
	NDArrayConverter cvt;
	cv::Mat image = cvt.toMat(image_);
	return cvt.toNDArray(segment_image(image, sigma, k, min_size));
}

static void init_ar() {
    Py_Initialize();
    import_array();
}

BOOST_PYTHON_MODULE(segment_felz) {
	init_ar();

	//initialize converters
	/*to_python_converter<cv::Mat,
		bcvt::matToNDArrayBoostConverter>();
		bcvt::matFromNDArrayBoostConverter();*/

	def("segment", segment);
}

int main(int argc, char **argv) {
	if (argc != 6) {
		std::cerr << "usage: " << argv[0] << " sigma k min image output" << std::endl;
		return 1;
	}
  
	float sigma = atof(argv[1]);
	int k = atoi(argv[2]);
	int min_size = atoi(argv[3]);
	
	cv::Mat image = cv::imread(argv[4]);
	//cv::resize(image, image, cv::Size(image.cols * 0.4, image.rows * 0.4));
	//int num_ccs; 

	std::cout << "Processing..." << std::endl;

	std::clock_t begin = std::clock();
	cv::Mat seg = segment_image(image, sigma, k, min_size); 
	std::clock_t end = std::clock();
	double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
	cv::imwrite(argv[5], seg);
	std::cout << "Time passed in seconds: " << elapsed_secs << std::endl;

	//std::cout << "Got " << num_ccs << " components." << std::endl;
	return 0;
}

