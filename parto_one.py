import time
import matplotlib.pyplot as plt
from DCT_lib import personal_dct2, dct2_library
import numpy as np


if __name__ == '__main__':
    # Dimensioni delle matrici NxN (da 50 a 900 con passo 50)
    matrix_dimensions = list(range(10, 1000, 50))

    times_homemade = []
    times_scipy = []

    for n in matrix_dimensions:
        print(n)
        # Creazione di una matrice random
        np.random.seed(5)
        matrix = np.random.uniform(low=0.0, high=255.0, size=(n, n))

        start_time = time.time()
        personal_dct2(matrix)
        times_homemade.append(time.time() - start_time)

        start_time = time.time()
        dct2_library(matrix)
        times_scipy.append(time.time() - start_time)

    # Dividiamo per 10^6 in modo da visualizzare le righe di comparazione vicino alla riga
    n3 = [n ** 3 / 1e5 for n in matrix_dimensions]
    n2_logn = [n ** 2 * np.log(n) / 1e8 for n in matrix_dimensions]

    plt.figure(figsize=(10, 6))
    plt.semilogy(matrix_dimensions, times_scipy, label='Scipy DCT2', color="#E53F0D")
    plt.semilogy(matrix_dimensions, n2_logn, label='n^2 * log(n)', color="#E53F0D", linestyle='dashed')
    plt.semilogy(matrix_dimensions, times_homemade, label='DCT2 homemade', color="tab:blue")
    plt.semilogy(matrix_dimensions, n3, label='n^3', color="tab:blue", linestyle='dashed')

    plt.xlabel('Dimensione N')
    plt.ylabel('Tempo di esecuzione in secondi')
    plt.title('Tempi di esecuzione della DCT2 al variare della dimensione N')
    plt.legend()
    plt.grid(True)

    plt.show()
