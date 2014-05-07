#include "Surface.h"

/*bool Surface::equals(const Surface & s) const {
	for (auto id: ids)
		if (s.ids.find(id) == s.ids.end())
			return false;
	return true;
}*/

/*Surface& Surface::operator=(const Surface & s) {
	ids = s.ids;
	return *this;
}*/

void Surface::addConnection(Connection * c) {
	//cm[s].push_back(pixel);
	cs.insert(c);
}

void Surface::merge(Connection * conn) {
	conn->a->ids.insert(conn->b->ids.begin(), conn->b->ids.end());
	conn->a->cs.erase(conn);
	conn->b->cs.erase(conn);

	for (auto c: conn->b->cs) {
		Surface * neighbour = c->b == conn->b ? c->a : c->b;
		auto it = conn->a->cs.find(new Connection(conn->a, neighbour));

		// it already exists
		if (it != conn->a->cs.end()) {
			Connection * c2 = *it;
			c2->merge(c->edge);
		}
		else {
			if (c->b == conn->b) c->b = conn->a;
			else c->a = conn->a;
		}
	}
}

std::ostream& operator<<(std::ostream& os, const Surface & s) {
	os << "[";
	for (auto id: s.ids)
		os << int(id) << ", ";
	return os << "]";
}

std::ostream& operator<<(std::ostream& os, const SurfaceSet & ss) {
	for (auto& s: ss)
		os << *(s.get()) << std::endl;
	return os;
}
