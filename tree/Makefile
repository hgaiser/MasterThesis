BIN = thesis

all:
	c++ tree.cpp `pkg-config --cflags --libs opencv` `pkg-config --cflags python` -std=c++11 -g -o $(BIN)

run:
	./$(BIN)
