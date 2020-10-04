
#include <iostream>
#include <cmath>
#include <vector>
#include <list>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>


namespace py = pybind11;

// init functions
std::vector<int> getNearest(std::vector<std::vector<int>> kMeans, std::vector<std::vector<int>> points);
float getDistance(std::vector<int> a, std::vector<int> b);
std::vector<std::vector<int>> updateKMeans(std::vector<std::vector<int>> kMeans, std::vector<std::vector<int>> points, std::vector<int> KsOfPoints);


// update K points by caculating the mean of the K-nearest points
std::vector<std::vector<int>> updateKMeans(std::vector<std::vector<int>> kMeans, std::vector<std::vector<int>> points, std::vector<int> KsOfPoints) {
        std::vector<std::vector<int>> newKMeans;
        for(int i=0; i<kMeans.size(); i++) {

                float sumR = 0.0;
                float sumG = 0.0;
                float sumB = 0.0;
                unsigned int total = 0;

                for(long unsigned int j=0; j<KsOfPoints.size(); j++) {
                        if(KsOfPoints[j] == i) {
                                sumR += points[j][0];
                                sumG += points[j][1];
                                sumB += points[j][2];
                                total++;
                        }
                }
                std::vector<int> newK;
                newK.push_back((int) (sumR / total));
                newK.push_back((int) (sumG / total));
                newK.push_back((int) (sumB / total));
                newKMeans.push_back(newK);
        }

        return newKMeans;
}


// get the nearest K of each point
std::vector<int> getNearest(std::vector<std::vector<int>> kMeans, std::vector<std::vector<int>> points) {
        std::vector<int> KsOfPoints;
        for(long unsigned int i=0; i<points.size(); i++) {

                int nearestK = 0;
                float shortestDistance = getDistance(kMeans[0], points[i]);
                for(long unsigned int j=1; j<kMeans.size(); j++) {
                        float currentDistance = getDistance(kMeans[j], points[i]);
                        if(currentDistance < shortestDistance) {
                                shortestDistance = currentDistance;
                                nearestK = j;
                        }
                }
                KsOfPoints.push_back(nearestK);
        }

        return KsOfPoints;
}


// calculate the euclidean distance between to vectors (size of 3,1 bc. of RGB channels)
float getDistance(std::vector<int> a, std::vector<int> b) {
        return sqrt(pow((b[0] - a[0]), 2) + pow((b[1] - a[1]), 2) + pow((b[2] - a[2]), 2));
}


// main function
py::array_t<int> segmentation(py::array_t<int> image, int k, int iterations) {
        auto imageBuf = image.mutable_unchecked<3>();
        int rows = imageBuf.shape(0);
        int cols = imageBuf.shape(1);
        int channels = 3;

        std::vector<std::vector<int>> indices;
        std::vector<std::vector<int>> points;
        int c = 0;
        for(int i=0; i<rows; i++) {
                for(int j=0; j<cols; j++) {

                        // create vector with a indices
                        std::vector<int> index;
                        index.push_back(i);
                        index.push_back(j);
                        indices.push_back(index);

                        // create vector with all rgb vectors (as coordinates)
                        std::vector<int> point;
                        for(int channel=0; channel<channels; channel++) {
                                point.push_back(imageBuf(i, j, channel));
                        }
                        points.push_back(point);

                        c++;
                }
        }


        // create inital K mean coordinates
        std::vector<std::vector<int>> kMeans;
        for(int i=0; i<k; i++) {
                std::vector<int> kMean;
                kMean.push_back(0 + (std::rand() % (255 - 0 + 1)));
                kMean.push_back(0 + (std::rand() % (255 - 0 + 1)));
                kMean.push_back(0 + (std::rand() % (255 - 0 + 1)));

                kMeans.push_back(kMean);
        }

       
        // fit K-Means on the data
        std::vector<int> KsOfPoints = getNearest(kMeans, points);
        for(int iteration=1; iteration<=iterations; iteration++) {
                kMeans = updateKMeans(kMeans, points, KsOfPoints);
                KsOfPoints = getNearest(kMeans, points);
        }


        //create segmentation image
        py::array_t<int> result = py::array_t<int>({rows, cols});
        auto resultBuf = result.mutable_unchecked<2>();

        for(long unsigned int i=0; i<indices.size(); i++) {
                int KofPoint = KsOfPoints[i];
                resultBuf(indices[i][0], indices[i][1]) = KofPoint;
        }

        return result;
}


PYBIND11_MODULE(segmentation, m) {
        m.doc() = "Creates a segmentation of an image using K-means.";
        m.def("segmentation", &segmentation, "Creates a segmentation of an image using K-means.");
}
