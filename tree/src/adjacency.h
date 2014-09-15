#ifndef ADJACENCY_H_
#define ADJACENCY_H_

class AdjacencyMatrix : public std::vector<Edge> {
public:
	AdjacencyMatrix(int n) {
		this->n = n;
		resize((n + 1)*n/2);
	}

	Edge& get(int i, int j) {
		int tmp = i > j ? j : i;
		j = i > j ? i : j;
		i = tmp;
		int ind = index(i, j);
		return operator[](ind);
	}

protected:
	int n;

	int index(int i, int j) {
		return (n * i) + j - ((i * (i+1)) / 2); 
	}
};

#endif

