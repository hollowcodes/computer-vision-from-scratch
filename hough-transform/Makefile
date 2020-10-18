all:
	c++ -Isrc -fPIC -I/usr/include/python3.7 -O3 -Wall -shared -std=c++11 -fPIC `python3 -m pybind11 --includes` houghTransform.cpp -o houghTransform`python3-config --extension-suffix`