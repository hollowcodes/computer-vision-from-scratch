
#include <cmath>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>


namespace py = pybind11;


py::array_t<float> edgeDetection(py::array_t<int> image) {

    auto imageBuf = image.mutable_unchecked<3>();
    int rows = imageBuf.shape(0);
    int cols = imageBuf.shape(1);

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

    py::array_t<float> result = py::array_t<float>({resultRows, resultCols});
    auto resultBuf = result.mutable_unchecked<2>();

    for(int k_i=0; k_i<resultRows; k_i++) {
        for(int k_j=0; k_j<resultCols; k_j++) {

            float sumX = 0;
            float sumY = 0;
            for(int i=0; i<kernelSize; i++) {        

                for(int j=0; j<kernelSize; j++) {
                    float px = imageBuf(i + k_i, j + k_j, 0) / 255.0;

                    sumX += px * kernelX[i][j];
                    sumY += px * kernelY[i][j];
                }
            }
            resultBuf(k_i, k_j) = sqrt(pow(sumX, 2.0) + pow(sumY, 2.0)) / sqrt(pow(1.9, 2.0) + pow(1.9, 2.0));
        }
    }

    return result;
}


PYBIND11_MODULE(edgeDetection, m) {
        m.doc() = "Edge detection.";
        m.def("edgeDetection", &edgeDetection, "Edge detection.");
}
