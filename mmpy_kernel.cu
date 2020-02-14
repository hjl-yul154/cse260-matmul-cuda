// Matrix multiply device code
#include <assert.h>
#include <math.h>
#include "types.h"
#include "utils.h"
using namespace std;

#define BLOCK_SIZE 32

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

    _DOUBLE_ c = 0;

    // Compute I,J of C
    int I = by * BLOCK_SIZE;
    int J = bx * BLOCK_SIZE;

    if (I + tx < N && J + ty < N) {
        for (int K = 0; K < N; K += BLOCK_SIZE) {
            // load I,K of A
            Ab[ty][tx] = A_ELEMENT(I + ty, K + tx);

            // load K,J of B
            Bb[ty][tx] = B_ELEMENT(K + ty, J + tx);

            __syncthreads();

            for (int k = 0; k < BLOCK_SIZE; ++k) {
                c += Ab[ty][k] * Bb[k][tx];
            }

            __syncthreads();
        }

        C_ELEMENT(I + ty, J + tx) = c;
    }
}
