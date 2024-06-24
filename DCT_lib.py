import numpy as np
from scipy.fft import dct


def dct_rows_only(input_matrix):
    N = input_matrix.shape[1]  # Numero di colonne

    j = np.arange(N)  # Indici fino a N
    i = j[:, None]  # Indici fino a N-1

    # Calcola gli argomenti del coseno per ciascuna coppia di indici (i, j).
    cos_matrix = np.cos(np.pi * (2 * i + 1) * j / (2 * N))

    # Matrice per il risultato della DCT riga per riga
    output_matrix = np.zeros_like(input_matrix, dtype=float)

    # Applica la DCT alle righe
    for x in range(input_matrix.shape[0]):  # Numero di righe
        for u in range(N):
            output_matrix[x, u] = np.sum(input_matrix[x] * cos_matrix[:, u])

    # Fattore di normalizzazione
    W = np.sqrt(2 / N) * np.ones(N)
    W[0] = np.sqrt(1 / N)

    # Applica i fattori di normalizzazione
    output_matrix *= W

    return output_matrix


def dct_columns_only(input_matrix):
    n = input_matrix.shape[0]  # Numero di righe

    j = np.arange(n)  # Indici fino a N
    i = j[:, None]  # Indici fino a N-1

    # Calcola gli argomenti del coseno per ciascuna coppia di indici (i, j).
    cos_matrix = np.cos(np.pi * (2 * i + 1) * j / (2 * n))

    # Matrice per il risultato della DCT colonna per colonna
    output_matrix = np.zeros_like(input_matrix, dtype=float)

    # Applica la DCT alle colonne
    for v in range(input_matrix.shape[1]):  # Numero di colonne
        for y in range(n):
            output_matrix[y, v] = np.sum(input_matrix[:, v] * cos_matrix[:, y])

    # Fattore di normalizzazione
    W = np.sqrt(2 / n) * np.ones(n)
    W[0] = np.sqrt(1 / n)

    # Applica i fattori di normalizzazione
    output_matrix *= W[:, None]

    return output_matrix


def personal_dct2(input_matrix):
    return dct_columns_only(dct_rows_only(input_matrix))


# Funzione DCT implementata da libreria esterna
# def scipy_dct(f):
#     return dct(f.T, 1, norm='ortho')
#
#
# # Funzione DCT2 implementata da libreria esterna
# def scipy_dct2(f):
#     return dct(f.T, 2, norm='ortho')

# Funzione DCT implementata da libreria esterna
def dct_library(f):
    return dct(f.T, norm='ortho')


# Funzione DCT2 implementata da libreria esterna
def dct2_library(f):
    return dct(dct(f.T, norm='ortho').T, norm='ortho')
