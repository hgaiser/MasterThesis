#ifndef SEGMENT_GRAPH
#define SEGMENT_GRAPH

#include "edge.h"
#include "cluster.h"

// threshold function
#define THRESHOLD(size, c) (c/size)

int find(std::vector<Cluster> & clusters, int x) {
	int y = x;
	while (y != clusters[y].id()) {
		y = clusters[y].id();
	}
	clusters[x].setId(y);
	return y;
}

void join(std::vector<Cluster> & clusters, int x, int y) {
	if (clusters[x].rank() > clusters[y].rank()) {
		clusters[y].setId(x);
		clusters[x].setSize(clusters[x].size() + clusters[y].size());
	} else {
		clusters[x].setId(y);
		clusters[y].setSize(clusters[y].size() + clusters[x].size());
		if (clusters[x].rank() == clusters[y].rank())
			clusters[y].setRank(clusters[y].rank() + 1);
	}
}

/*
 * Segment a graph
 *
 * Returns a disjoint-set forest representing the segmentation.
 *
 * num_vertices: number of vertices in graph.
 * num_edges: number of edges in graph
 * edges: array of edges.
 * c: constant for treshold function.
 */
std::vector<Cluster> segment_graph(int num_vertices, int num_edges, std::vector<Edge> & edges, float c, int & num) { 
	// sort edges by weight
	std::sort(edges.begin(), edges.begin() + num_edges);

	// make a disjoint-set forest
	std::vector<Cluster> clusters;
	clusters.resize(num_vertices);

	// init thresholds
	for (int i = 0; i < num_vertices; i++) {
		clusters[i].setId(i);
		clusters[i].setThreshold(THRESHOLD(1,c));
	}

	// for each edge, in non-decreasing weight order...
	for (int i = 0; i < num_edges; i++) {
		Edge & edge = edges[i];
    
		// components conected by this edge
		int a = find(clusters, edge.first());
		int b = find(clusters, edge.second());
		if (a != b) {
			if ((edge.weight() <= clusters[a].threshold()) &&
				(edge.weight() <= clusters[b].threshold())) {
				join(clusters, a, b);
				a = find(clusters, a);
				clusters[a].setThreshold(edge.weight() + THRESHOLD(clusters[a].size(), c));
				num--;
			}
		}
	}

	return clusters;
}

#endif
