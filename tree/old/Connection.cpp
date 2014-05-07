#include "Connection.h"
#include "Surface.h"

Connection::Connection(Surface * a, Surface * b): 
		a(a), b(b), median(0), changed(true)
{}

std::ostream& operator<<(std::ostream& os, const Connection & c) {
	os << *c.a << " -> " << *c.b << ": ";
	//for (auto pixel: c.edge)
	//	os << int(pixel) << ", ";
	return os;
}

void Connection::add(uint8_t value) {
	changed = true;
	if (edge.size() == 0) edge.push_back(value);
}

uint8_t Connection::computeMedian() {
	if (changed) {
		std::sort(edge.begin(), edge.end());
		median = edge[edge.size() / 2];
	}

	changed = false;
	return median;
}

void Connection::merge(std::vector<uint8_t> other_edge) {
	changed = true;
	std::vector<uint8_t> tmp;
	tmp.reserve(edge.size() + other_edge.size());
	std::merge(edge.begin(), edge.end(), other_edge.begin(), other_edge.end(), std::back_inserter(tmp));
	edge.swap(tmp);
}

std::ostream& operator<<(std::ostream& os, const ConnectionSet & cs) {
	for (auto& c: cs)
		os << *c << std::endl;
	return os;
}
