// Matrix multiply device code
#include <assert.h>
#include <math.h>
#include "types.h"
#include "utils.h"
using namespace std;

#define BLOCK_SIZE 32
#if BLOCK_SIZE % BLOCKDIM_X || BLOCK_SIZE % BLOCKDIM_Y
#error BLOCK_SIZE must be multiple of blockDim
#endif
// Number of thread blocks in matrix block
#define X_SUB (BLOCK_SIZE / BLOCKDIM_X)
#define Y_SUB (BLOCK_SIZE / BLOCKDIM_Y)

#define MAT(mat, N, i, j) (mat[(i)*N + (j)])
#define MAT_PADDED(mat, N, i, j) ((i) < N && (j) < N ? MAT(mat, N, i, j) : 0)
#define A_ELEMENT(i, j) MAT_PADDED(A, N, i, j)
#define B_ELEMENT(i, j) MAT_PADDED(B, N, i, j)
#define C_ELEMENT(i, j) MAT(C, N, i, j)

__global__ void matMul(int N, _DOUBLE_* C, _DOUBLE_* A, _DOUBLE_* B) {
    __shared__ _DOUBLE_ Ab[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ _DOUBLE_ Bb[BLOCK_SIZE][BLOCK_SIZE];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    _DOUBLE_ c[Y_SUB][X_SUB] = {0}; // Zero initialize the whole array

    // Compute I0,J0 of C
    int I0 = by * BLOCK_SIZE;
    int J0 = bx * BLOCK_SIZE;

    for (int K = 0; K < N; K += BLOCK_SIZE) {
        for (int i = 0; i < BLOCK_SIZE; i += BLOCKDIM_Y) {
            for (int j = 0; j < BLOCK_SIZE; j += BLOCKDIM_X) {
                Ab[ty + i][tx + j] = A_ELEMENT(I0 + ty + i, K + tx + j);
            }
        }

        for (int i = 0; i < BLOCK_SIZE; i += BLOCKDIM_Y) {
            for (int j = 0; j < BLOCK_SIZE; j += BLOCKDIM_X) {
                Bb[ty + i][tx + j] = B_ELEMENT(K + ty + i, J0 + tx + j);
            }
        }

        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; ++k) {
            for (int i = 0; i < Y_SUB; ++i) {
                for (int j = 0; j < X_SUB; ++j) {
                    c[i][j] += Ab[ty + i * BLOCKDIM_Y][k] * Bb[k][tx + j * BLOCKDIM_X];
                }
            }
        }

        __syncthreads();
    }

    for (int i = 0; i < Y_SUB; ++i) {
        for (int j = 0; j < X_SUB; ++j) {
            if (I0 + ty + i * BLOCKDIM_Y < N && J0 + tx + j * BLOCKDIM_X < N) {
                C_ELEMENT(I0 + ty + i * BLOCKDIM_Y, J0 + tx + j * BLOCKDIM_X) = c[i][j];
            }
        }
    }
}
