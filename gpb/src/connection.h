#pragma once

class Connection
{
public:
	Connection(int a_, int b_, float sim) :
		a(a_), b(b_), similarity(sim)
	{
	}

	friend std::ostream& operator<<(std::ostream &out, Connection & c) {
		return out << "<a: " << c.a << "; b: " << c.b << "; sim: " << c.similarity << ">";
	}

	bool operator<(const Connection & rhs) const {
		return rhs.similarity > similarity;
	}

	int a;
	int b;
	float similarity;
};

