import numpy as np
import math


# Function to compute ci and cj
def compute_c(p, m):
    # where if i=0
    #   ci= 1/sqrt(m)
    # else ci= sqrt(2)/sqrt(m)
    if p == 0:
        cp = 1/np.sqrt(m)
    else:
        cp = np.sqrt(2) / np.sqrt(m)
    return cp


def compute_dct(mtrx, dim):
    # dct[i][j] =
    #   ci * cj * {sum(k=0 to m-1) sum(l=0 to n-1) matrix[k][l] * cos[(2*k+1) *i*pi/2*m] * cos[(2*l+1) *j*pi/2*n]}

    matrix_dct = []

    for i in range(dim):
        for j in range(dim):
            ci = compute_c(i, dim)
            cj = compute_c(j, dim)

            for k in range(dim):
                for l in range(dim):

                    dct = ci * cj * (
                        np.sum([dim-1], initial=0) *
                        np.sum([dim-1], initial=0) *
                        mtrx[k][l] *
                        math.cos((2*k+1) * i * math.pi/(2*dim)) *
                        math.cos((2*l+1) * j * math.pi/(2*dim))
                    )
                    matrix_dct.append(dct)

    # return tempo


if __name__ == '__main__':
    MAX_N = 100
    tempi = []

    for n in range(10, MAX_N, 10):
        matrix = np.arange(n * n).reshape(n, n)
        print(matrix.shape)
        print(matrix)

        tempi.append(compute_dct(matrix, n))
