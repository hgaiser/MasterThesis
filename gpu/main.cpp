#include "conversion.h"
#include <boost/python.hpp>

#include "segmentationTree.h"

using namespace boost::python;


PyObject * computeSegmentationLevels(PyObject * image_) {
	NDArrayConverter cvt;
	cv::Mat image = cvt.toMat(image_);
	std::vector<cv::Mat> levels = computeSegmentationLevels_(image);
	PyArrayObject * arr;
	int dims[1] = { levels.size() };
	arr = (PyArrayObject *) PyArray_FromDims(1, dims, 'O'); 
	PyObject ** data = (PyObject **)arr->data;
	for (int i = 0; i < levels.size(); i++)
		data[i] = cvt.toNDArray(levels[i]);

	return PyArray_Return(arr);
}

static void init_ar() {
	Py_Initialize();
	import_array();
}

BOOST_PYTHON_MODULE(gpu_segmentation) {
	init_ar();

	def("startCuda", startCuda);
	def("stopCuda", stopCuda);
	def("computeSegmentationLevels", computeSegmentationLevels);
}

int main(int argc, char * argv[]) {
}
