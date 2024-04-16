#include <memory>
#include <vector>
#include <cmath>

//number of parallel instructions
constexpr int pI = 4;

typedef double double4_t __attribute__ ((vector_size (4 * sizeof(double))));

void matmulTFullParallelization(int ny, int nx, double* matrix, double* result) {

    int nxP = (nx + pI - 1) / pI;    

    std::vector<double4_t> matrixPadded(ny * nxP);
    #pragma omp parallel for
    for(int k = 0; k < ny; k++) {
        for(int l = 0; l < nxP; l++) {
            for(int m = 0; m < pI; m++) {                      
                int i = l * pI + m;  
                matrixPadded[nxP * k + l][m] = i <  nx ? matrix[nx * k + l * pI + m] : 0.0;
            }
        }
    }

    #pragma omp parallel for schedule(static,1)
    for(int k = 0; k < ny; k++) {
        for(int l = k; l < ny; l++) {
            double4_t vectorSum = {0.0, 0.0, 0.0, 0.0};
            for(int m = 0; m < nxP; m++) {                        
                vectorSum += matrixPadded[k * nxP + m] * matrixPadded[l * nxP + m];        
            }

            for(int m = 0; m < pI; m++) {
                result[k * ny + l] += vectorSum[m];   
            }
        }
    }
}

void matmulTMultiThread(int ny, int nx, double* matrix, double* result) {

    #pragma omp parallel for schedule(static,1)
    for(int k = 0; k < ny; k++) {        
        for(int l = k; l < ny; l++) {
            for(int m = 0; m < nx; m++) {
                result[k * ny + l] += matrix[k * nx + m] * matrix[l * nx + m];   
            }
        }
    }
}


void matmulTVectorizationParallel(int ny, int nx, double* matrix, double* result) {

    int nxP = (nx + pI - 1) / pI;    

    std::vector<double4_t> matrixPadded(ny * nxP);
    for(int k = 0; k < ny; k++) {
        for(int l = 0; l < nxP; l++) {
            for(int m = 0; m < pI; m++) {                      
                int i = l * pI + m;  
                matrixPadded[nxP * k + l][m] = i <  nx ? matrix[nx * k + l * pI + m] : 0.0;
            }
        }
    }

    for(int k = 0; k < ny; k++) {
        for(int l = k; l < ny; l++) {
            double4_t vectorSum = {0.0, 0.0, 0.0, 0.0};
            for(int m = 0; m < nxP; m++) {                        
                vectorSum += matrixPadded[k * nxP + m] * matrixPadded[l * nxP + m];        
            }

            for(int m = 0; m < pI; m++) {
                result[k * ny + l] += vectorSum[m];   
            }
        }
    }
}

void matmulTInstrunctionParallel(int ny, int nx, double* matrix, double* result) {

    int nxP = (nx + pI - 1) / pI;
    int nxTot = nxP * pI;

    std::vector<double> matrixPadded(ny * nxTot, 0.0);
    for(int k = 0; k < ny; k++) {
        for(int l = 0; l < nx; l++) {
            matrixPadded[nxTot * k + l] = matrix[nx * k + l];            
        }
    }

    for(int k = 0; k < ny; k++) {
        for(int l = k; l < ny; l++) {
            std::vector<double> pS(pI);
            for(int m = 0; m < nxP; m++) {
                for(int n = 0; n < pI; n++) {
                    pS[n] += matrixPadded[k * nxTot + m * pI +n] * matrixPadded[l * nxTot + m * pI + n];                       
                }
            }

            for(const auto& elem : pS) {
                result[k * ny + l] += elem;   
            }
        }
    }
}

void matmulT(int ny, int nx, double* matrix, double* result) {

    for(int k = 0; k < ny; k++) {
        for(int l = k; l < ny; l++) {
            for(int m = 0; m < nx; m++) {
                result[k * ny + l] += matrix[k * nx + m] * matrix[l * nx + m];   
            }
        }
    }
}

void normalize(int ny, int nx, double *result) {

    std::vector<double> rowAverage(ny);
    std::vector<double> rowSquare(ny);

    for(int k = 0; k < ny; k++) {        
        for(int l = 0; l < nx; l++) {
            rowAverage[k] += result[k * nx + l];
        }
        rowAverage[k] /= nx;
    }

    for(int k = 0; k < ny; k++) {        
        for(int l = 0; l < nx; l++) {
            result[k * nx + l] -= rowAverage[k];
        }        
    }

    for(int k = 0; k < ny; k++) {        
        for(int l = 0; l < nx; l++) {
            rowSquare[k] += result[k * nx + l] * result[k * nx + l];
        }

        rowSquare[k] = sqrt(rowSquare[k]);        
    }
    
    for(int k = 0; k < ny; k++) {        
        for(int l = 0; l < nx; l++) {
            result[k * nx + l] /= rowSquare[k];
        }        
    }
}


/*
This is the function you need to implement. Quick reference:
- input rows: 0 <= y < ny
- input columns: 0 <= x < nx
- element at row y and column x is stored in data[x + y*nx]
- correlation between rows i and row j has to be stored in result[i + j*ny]
- only parts with 0 <= j <= i < ny need to be filled
*/
void correlate(int ny, int nx, const float *data, float *result) {

    std::unique_ptr<double[]> matrix = std::make_unique<double[]>(nx * ny);    
    for(int k = 0; k < ny ; k++) {
        for(int l = 0; l < nx; l++) {
            matrix[k * nx + l] = data[k * nx + l];
        }
    }

    std::unique_ptr<double[]> doubleResult = std::make_unique<double[]>(ny * ny);
    normalize(ny, nx, matrix.get());
    matmulTFullParallelization(ny, nx, matrix.get(), doubleResult.get());
    for(int elem = 0; elem < ny * ny; elem++) {
        result[elem] = static_cast<float>(doubleResult[elem]);
    }
}
