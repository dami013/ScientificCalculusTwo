import time
import matplotlib.pyplot as plt
from DCT_lib import personal_dct2, dct2_library
import numpy as np


if __name__ == '__main__':
    # Dimensioni delle matrici NxN (da 200 a 1000 con passo 50)
    matrix_dimensions = list(range(200, 1000, 50))

    times_homemade = []
    times_scipy = []

    for n in matrix_dimensions:
        print(n)
        # Creazione di una matrice random
        np.random.seed(5)
        matrix = np.random.uniform(low=0.0, high=255.0, size=(n, n))

        start_time = time.time()
        personal_dct2(matrix)
        total_time = time.time() - start_time
        times_homemade.append(total_time)
        print("Homemade "+str(n)+" tempo: "+str(total_time))

        start_time2 = time.time()
        dct2_library(matrix)
        total_time2 = time.time() - start_time2
        times_scipy.append(total_time2)
        print("Scipy "+str(n)+" tempo: "+str(total_time2))

    # Dividiamo per 10^6 in modo da visualizzare le righe di comparazione vicino alla riga
    n3 = [n ** 3 / 1e5 for n in matrix_dimensions]
    n2_logn = [n ** 2 * np.log(n) / 1e8 for n in matrix_dimensions]

    plt.figure(figsize=(10, 6))
    plt.semilogy(matrix_dimensions, times_scipy, label='Scipy DCT2', color="green")
    plt.semilogy(matrix_dimensions, n2_logn, label='n^2 * log(n)', color="green", linestyle='dashed')
    plt.semilogy(matrix_dimensions, times_homemade, label='DCT2 homemade', color="tab:blue")
    plt.semilogy(matrix_dimensions, n3, label='n^3', color="tab:blue", linestyle='dashed')

    plt.xlabel('Dimensione N')
    plt.ylabel('Tempo di esecuzione in secondi')
    plt.title('Tempi di esecuzione della DCT2 al variare della dimensione N')
    plt.legend()
    plt.grid(True)

    # Salva l'immagine
    plt.savefig('times-plot.png', dpi=300, bbox_inches='tight')

    plt.show()
