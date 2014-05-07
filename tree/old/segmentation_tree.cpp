#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <unordered_map>
#include <unordered_set>
#include <ctime>

#include "Surface.h"
#include "Connection.h"

Surface* addSurface(SurfaceSet & ss, uint8_t id) {
	//std::cout<<"addSurface"<<std::endl;
	std::unique_ptr<Surface> s_(new Surface(id));
	//auto it = ss.find(s_);
	//if (it != ss.end())
	//	return it->get();
	//else
		return ss.insert(std::move(s_)).first->get();
}

Connection* addConnection(ConnectionSet & cs, Surface * s1, Surface * s2) {
	//std::cout<<"addConnection"<<std::endl;
	std::unique_ptr<Connection> c_(new Connection(s1, s2));
	//auto it = cs.find(c_);
	//if (it != cs.end())
	//	return it->get();
	//else
		return cs.insert(std::move(c_)).first->get();
}

void calculateConnections(cv::Mat segmentation, cv::Mat edge) {
	SurfaceSet ss;
	ConnectionSet cs;

	cv::Mat nonzero;
	cv::findNonZero(segmentation == 0, nonzero);
	for (int i = 0; i < nonzero.total(); i++) {
		cv::Point p = nonzero.at<cv::Point>(i);
		if (p.x <= 0 || p.x >= segmentation.cols - 1 || p.y <= 0 || p.y >= segmentation.rows - 1)
			continue;

	    if (segmentation.at<uint8_t>(p.y-1, p.x) > 0 && segmentation.at<uint8_t>(p.y+1, p.x) > 0 && segmentation.at<uint8_t>(p.y-1, p.x) != segmentation.at<uint8_t>(p.y+1, p.x)) {
	    	Surface * s1 = addSurface(ss, segmentation.at<uint8_t>(p.y-1, p.x));
	    	Surface * s2 = addSurface(ss, segmentation.at<uint8_t>(p.y+1, p.x));
	    	Connection * c = addConnection(cs, s1, s2);

	    	c->add(edge.at<uint8_t>(p.y, p.x));
			s1->addConnection(c);
			s2->addConnection(c);

	        //if (segmented_edge) segmented_edge->at<uint8_t>(p.y, p.x) = edge.at<uint8_t>(p.y, p.x);
	    }
	    else if (segmentation.at<uint8_t>(p.y, p.x-1) > 0 && segmentation.at<uint8_t>(p.y, p.x+1) > 0 && segmentation.at<uint8_t>(p.y, p.x-1) != segmentation.at<uint8_t>(p.y, p.x+1)) {
			Surface * s1 = addSurface(ss, segmentation.at<uint8_t>(p.y, p.x-1));
	    	Surface * s2 = addSurface(ss, segmentation.at<uint8_t>(p.y, p.x+1));
	    	Connection * c = addConnection(cs, s1, s2);

	    	c->add(edge.at<uint8_t>(p.y, p.x));
			s1->addConnection(c);
			s2->addConnection(c);
			//if (segmented_edge) segmented_edge->at<uint8_t>(p.y, p.x) = edge.at<uint8_t>(p.y, p.x);
	    }
	}

	std::cout << "size: " << ss.size() << std::endl;

	/*while (cs.size() != 1) {
		uint8_t min_median = 255;
		Connection * min_conn;
		for (auto& c: cs) {
			uint8_t median = c->computeMedian();
			if (median < min_median) {
				min_median = median;
				min_conn = c.get();
			}
		}
		//std::cout << "ConnectionSet: " << cs << std::endl;
		//std::cout << "MinConn: " << *min_conn << std::endl;
		//if (cs.find(min_conn) == cs.end()) std::cout << cs << std::endl;
		cs.erase(std::unique_ptr<Connection>(min_conn));
		Surface::merge(min_conn);
	}*/

	//std::cout << cs << std::endl;
	//std::cout << *min_conn << std::endl;

	//std::cout << "Min Edge: " << *min_conn << " with edge: " << int(min_edge) << std::endl;
}

int main(int argc, char * argv[]) {
	//cv::namedWindow("Segmentation", cv::WINDOW_NORMAL);
	//cv::namedWindow("Edge", cv::WINDOW_NORMAL);
	//cv::namedWindow("Segmented Edge", cv::WINDOW_NORMAL);
	//cv::startWindowThread();
	cv::Mat segmentation = cv::imread("segmentation.png", cv::IMREAD_GRAYSCALE);
	cv::Mat edge = cv::imread("edge.png", cv::IMREAD_GRAYSCALE);
	cv::Mat segmented_edge = cv::Mat::zeros(segmentation.size(), CV_8UC1);
	//cv::imshow("Segmentation", segmentation);
	//cv::imshow("Edge", edge);

	std::clock_t begin = std::clock();

	/*ConnectionStrengthMap csm = calculateConnections(segmentation, edge, &segmented_edge);

	ConnectionMedianMap cmm;
	const Connection * min_conn = NULL;
	uint8_t min_edge = 255;
	for (ConnectionStrengthMap::iterator it = csm.begin(); it != csm.end(); ++it) {
		std::sort(it->second.begin(), it->second.end());
		cmm[it->first] = it->second[it->second.size() / 2];
		if (cmm[it->first] < min_edge) {
			min_edge = cmm[it->first];
			min_conn = &it->first;
		}
	}

	for (ConnectionMedianMap::iterator it = cmm.begin(); it != cmm.end(); ++it) {
		std::cout << it->first << " : " << int(it->second) << std::endl;
	}

	std::cout << *min_conn << " : " << int(min_edge) << std::endl;*/

	calculateConnections(segmentation, edge);

	std::clock_t end = std::clock();
	double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
	std::cout << "Times passed in seconds: " << elapsed_secs << std::endl;
	//cv::imshow("Segmented Edge", segmented_edge);

	//cv::waitKey();
	return 0;
}
