
#include <cmath>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>


#define M_PI 3.14159265358979323846

namespace py = pybind11;
using pybind11_arr = pybind11::detail::unchecked_mutable_reference<int, 3>;


py::array_t<int> edgeDetect(pybind11_arr image, int rows, int cols) {
    // sobel filter for x-direction
    const float kernelX[3][3] = {{1.0, 2.0, 1.0},
                                {0.0, 0.0, 0.0}, 
                                {-1.0, -2.0, -1.0}};
                          
    // sobel filter for y-direction
    const float kernelY[3][3] = {{1.0, 0.0, -1.0},
                                {2.0, 0.0, -2.0}, 
                                {1.0, 0.0, -1.0}};
    
    const int kernelSize = 3;

    int resultRows = rows - kernelSize + 1;
    int resultCols = cols - kernelSize + 1;

    py::array_t<int> result = py::array_t<int>({resultRows, resultCols});
    auto resultBuf = result.mutable_unchecked<2>();

    for(int k_i=0; k_i<resultRows; k_i++) {
        for(int k_j=0; k_j<resultCols; k_j++) {

            float sumX = 0;
            float sumY = 0;
            for(int i=0; i<kernelSize; i++) {        

                for(int j=0; j<kernelSize; j++) {
                    float px = image(i + k_i, j + k_j, 0) / 255.0;

                    sumX += px * kernelX[i][j];
                    sumY += px * kernelY[i][j];
                }
            }
            resultBuf(k_i, k_j) = (int) std::round(sqrt(pow(sumX, 2.0) + pow(sumY, 2.0)) / sqrt(pow(1.8, 2.0) + pow(1.8, 2.0)));
        }
    }

    return result;
}


py::array_t<int> houghTransform(py::array_t<int> image, int angleStep) {
    auto imageBuf = image.mutable_unchecked<3>();
    int height = imageBuf.shape(0);
    int width = imageBuf.shape(1);

    py::array_t<int> edgeMatrix = edgeDetect(imageBuf, height, width);
    auto edgeMatrixBuf = edgeMatrix.mutable_unchecked<2>();

    int distanceAxis = 2 * sqrt(pow((float) height, 2.0) + pow((float) width, 2.0));
    int angleAxis = 180 * angleStep;
    
    int distanceDim = (int) distanceAxis / 2;

    py::array_t<int> votingMatrix = py::array_t<int>({distanceAxis, angleAxis});
    auto votingMatrixBuf = votingMatrix.mutable_unchecked<2>();

    // fill voting matrices with zeros
    for(int i=0; i<distanceAxis; i++) {
        for(int j=0; j<angleAxis; j++) {
            votingMatrixBuf(i, j) = 0;
        }
    }

    // vote
    for(int x=0; x<edgeMatrixBuf.shape(0); x++) {
        for(int y=0; y<edgeMatrixBuf.shape(1); y++) {
            
            if(edgeMatrixBuf(x, y) != 0) {

                float theta;
                float ro;

                for(int thetaIdx=0; thetaIdx<=angleAxis; thetaIdx++) {
                    theta = (float) (thetaIdx);
                    theta = (theta / (float) angleStep) * (M_PI / 180);

                    ro = distanceDim + std::round((x * cos(theta)) + (y * sin(theta)));
                    py::print(theta);

                    votingMatrixBuf(ro, thetaIdx) += 1;
                }

            }
        }
    }

    return votingMatrix;
}


PYBIND11_MODULE(houghTransform, m) {
        m.doc() = "Line detection.";
        m.def("houghTransform", &houghTransform, "Line detection.");
}
