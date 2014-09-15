#ifndef CLUSTER_H_
#define CLUSTER_H_

class Cluster {
public:
	Cluster() {
		setSize(1);
		setRank(0);
	}

	inline float threshold() const { return threshold_; }
	inline uint32_t size() const { return size_; }
	inline uint32_t rank() const { return rank_; }
	inline uint32_t id() const { return id_; }

	inline void setThreshold(float t) { threshold_ = t; }
	inline void setSize(int s) { size_ = s; }
	inline void setRank(int r) { rank_ = r; }
	inline void setId(int id) { id_ = id; }

private:
	float threshold_;
	uint32_t size_;
	uint32_t rank_;
	uint32_t id_;
};

#endif
