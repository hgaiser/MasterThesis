#ifndef EDGE_H_
#define EDGE_H_

class Edge {
public:
	Edge(uint32_t a, uint32_t b, float w) : a_(a), b_(b), weight_(w) {}
	Edge() {}

	inline float weight() { return weight_; }
	inline uint32_t first() { return a_; }
	inline uint32_t second() { return b_; }

	inline void setFirst(int a) { a_ = a; }
	inline void setSecond(int b) { b_ = b; }
	inline void setWeight(float w) { weight_ = w; }

private:
	uint32_t a_;
	uint32_t b_;
	float weight_;
};

#endif
