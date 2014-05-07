/*
 * Connection.h
 *
 *  Created on: Mar 19, 2014
 *      Author: hans
 */

#ifndef CONNECTION_H_
#define CONNECTION_H_

#include <vector>
#include <algorithm>
#include <memory>
#include <stdint.h>
#include <unordered_set>
#include <ostream>

class Surface;

class Connection {
public:
	Surface * a;
	Surface * b;
	std::vector<uint8_t> edge;
	uint8_t median;
	bool changed;

	Connection(Surface * a, Surface * b);

	friend std::ostream& operator<<(std::ostream& os, const Connection & c);

	void add(uint8_t value);
	void merge(std::vector<uint8_t> other_edge);
	uint8_t computeMedian();
};

struct ConnectionHash {
	size_t operator() (const std::unique_ptr<Connection> & c) const {
		size_t hash = 0;
		if (c) {
			hash = std::hash<Surface *>()(c->a) ^ std::hash<Surface *>()(c->b);
		}

		return hash;
	}
};

struct ConnectionEqual {
    bool operator() (const std::unique_ptr<Connection> & lhs, const std::unique_ptr<Connection> & rhs) const {
    	if (lhs == rhs)
    		return true;
    	if (!lhs || !rhs)
    		return false;

    	return (lhs->a == rhs->a && lhs->b == rhs->b) || (lhs->b == rhs->a && lhs->a == rhs->b);
   	}
};

typedef std::unordered_set<std::unique_ptr<Connection>, ConnectionHash, ConnectionEqual> ConnectionSet;

std::ostream& operator<<(std::ostream& os, const ConnectionSet & cs);

#endif /* CONNECTION_H_ */
