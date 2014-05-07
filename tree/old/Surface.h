/*
 * Surface.h
 *
 *  Created on: Mar 19, 2014
 *      Author: hans
 */

#ifndef SURFACE_H_
#define SURFACE_H_

#include <iostream>
#include <unordered_set>
#include <unordered_map>
#include <vector>

#include "Connection.h"

class Surface {
public:
	std::unordered_set<Connection *> cs;
	std::unordered_set<uint8_t> ids;
	size_t hash;

	Surface(int id) {
		ids.insert(id);
		hash = id;
	};

	//bool equals(const Surface & s) const;
	//Surface& operator=(const Surface & s);

	void addConnection(Connection * c);
	static void merge(Connection * conn);

	friend std::ostream& operator<<(std::ostream& os, const Surface & s);
};

struct SurfaceHash {
	size_t operator() (const std::unique_ptr<Surface> & s) const {
		return s->hash;
	}
};

struct SurfaceEqual {
    bool operator() (const std::unique_ptr<Surface> & a, const std::unique_ptr<Surface> & b) const {
		//std::cout<<"SurfaceEqual"<<std::endl;
    	if (a == b)
    		return true;
    	if (!a || !b)
    		return false;

    	for (auto id: a->ids)
    		if (b->ids.find(id) == b->ids.end())
    			return false;
    	return true;
   	}
};

typedef std::unordered_set<std::unique_ptr<Surface>, SurfaceHash, SurfaceEqual> SurfaceSet;
std::ostream& operator<<(std::ostream& os, const SurfaceSet & ss);

#endif /* SURFACE_H_ */
