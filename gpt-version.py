import numpy as np
from scipy.fftpack import dct
import time
import matplotlib.pyplot as plt

# Homemade DCT2 implementation
def create_dct_matrix(N):
    dct_matrix = np.zeros((N, N))
    factor = np.pi / (2.0 * N)
    for i in range(N):
        for j in range(N):
            if i == 0:
                dct_matrix[i, j] = 1.0 / np.sqrt(N)
            else:
                dct_matrix[i, j] = np.sqrt(2.0 / N) * np.cos(factor * (2 * j + 1) * i)
    return dct_matrix

def dct2_manual(x):
    N = x.shape[0]
    dct_matrix_N = create_dct_matrix(N)
    return np.dot(dct_matrix_N, np.dot(x, dct_matrix_N.T))

# Library DCT2 implementation
def dct2_library(x):
    return dct(dct(x, axis=0, norm='ortho'), axis=1, norm='ortho')

# Measure and plot the execution times
Ns = [2**i for i in range(1, 10)]  # Sizes from 2x2 to 512x512
times_manual = []
times_library = []

for N in Ns:
    x = np.random.random((N, N))

    # Measure time for homemade DCT2
    start_time = time.time()
    dct2_manual(x)
    times_manual.append(time.time() - start_time)

    # Measure time for library DCT2
    start_time = time.time()
    dct2_library(x)
    times_library.append(time.time() - start_time)

# Theoretical complexities
times_n3 = [N**3 for N in Ns]
times_n2_log_n = [N**2 * np.log2(N) for N in Ns]

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(Ns, times_manual, 'o-', label='DCT2 created', color='green')
plt.plot(Ns, times_library, 's-', label='Library DCT2', color='red')
plt.plot(Ns, times_n3, '--', label='n^3', color='green')
plt.plot(Ns, times_n2_log_n, '--', label='n^2 * log(n)', color='red')
plt.yscale('log')
plt.xlabel('Dimensione N')
plt.ylabel('Tempo di esecuzione in secondi')
plt.title('Tempi di esecuzione della DCT2 al variare della dimensione N')
plt.legend()
plt.grid(True)
plt.show()