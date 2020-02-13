// Matrix multiply device code
#include <assert.h>
#include <math.h>
#include "utils.h"
#include "types.h"
using namespace std;

#define BLOCK_SIZE 32

#define MAT_ELEMENT(mat, N, i, j) ((i)<N && (j)<N ? mat[(i)*N+(j)] : 0)
#define A_ELEMENT(i, j) MAT_ELEMENT(A, N, i, j)
#define B_ELEMENT(i, j) MAT_ELEMENT(B, N, i, j)

__global__ void matMul(int N, _DOUBLE_* C, _DOUBLE_* A, _DOUBLE_* B) {
	__shared__ _DOUBLE_ Ab[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ _DOUBLE_ Bb[BLOCK_SIZE][BLOCK_SIZE];

	int bx = blockIdx.x, by = blockIdx.y;
	int tx = threadIdx.x, ty = threadIdx.y;

	_DOUBLE_ c = 0;

	// Compute I,J of C
	int I = bx * BLOCK_SIZE;
	int J = by * BLOCK_SIZE;

	for (int K = 0; K < N; K += BLOCK_SIZE) {
		// load I,K of A
		Ab[tx][ty] = A_ELEMENT(I + tx, K + ty);

		// load K,J of B
		Bb[tx][ty] = B_ELEMENT(K + tx, J + ty);

		__syncthreads();

		for (int k = 0; k < BLOCK_SIZE; ++k) {
			c += Ab[tx][k] * Bb[k][ty];
		}

		__syncthreads();
	}

	if (I + tx < N && J + ty < N) {
		C[(I + tx) * N + (J + ty)] = c;
	}
}
