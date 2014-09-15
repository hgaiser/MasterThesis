#ifndef ADJACENCY_H_
#define ADJACENCY_H_

#include <vector>

class AdjacencyMatrix : public std::vector<bool> {
public:
	AdjacencyMatrix(int n) {
		this->n = n;
		assign((n + 1)*n/2, false);
	}

	reference get(int i, int j) {
		return at(index(i, j));
	}

protected:
	int n;

	int index(int i, int j) {
		int tmp = i > j ? j : i;
		j = i > j ? i : j;
		i = tmp;
		return (n * i) + j - ((i * (i+1)) / 2);
	}
};

#endif

