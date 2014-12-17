/*
Copyright (C) 2006 Pedro Felzenszwalb

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA
*/

#include <cstdio>
#include <cstdlib>
#include "image.h"
#include "misc.h"
#include "pnmfile.h"
#include "segment-image.h"
#include "conversion.h"
#include <boost/python.hpp>

using namespace boost::python;

cv::Mat segment_image(cv::Mat & im, float sigma, int k, int min_size) {
	int w = im.cols;
	int h = im.rows;
	image<rgb> input(w, h);

	for (int i = 0; i < h; i++) {
		for (int j = 0; j < w; j++) {
			cv::Vec3b p = im.at<cv::Vec3b>(i, j);
			input.data[i*w + j].b = p[0];
			input.data[i*w + j].g = p[1];
			input.data[i*w + j].r = p[2];
		}
	}

	int num_ccs;
	image<uint16_t> * seg = segment_image(&input, sigma, k, min_size, &num_ccs);

	cv::Mat output(h, w, CV_16UC1);
	for (int i = 0; i < h; i++) {
		for (int j = 0; j < w; j++) {
			output.at<uint16_t>(i, j) = seg->data[i*w + j];
		}
	}
	delete seg;

	return output;
}

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

	def("segment", segment);
}

