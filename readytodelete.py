import numpy as np


def dct2_home_made(input_matrix):
    N = input_matrix.shape[0]

    j = np.arange(N)  # indice fino ad N
    i = j[:, None]  # indice fino ad N-1

    # calcola gli argomenti del coseno per ciascuna coppia di indici (i, j).
    cos_matrix = np.cos(np.pi * (2 * i + 1) * j / (2 * N))

    # Matrice per i risultati intermedi della trasformazione
    temp_matrix = np.zeros_like(input_matrix, dtype=float)

    # Applica la DCT alle righe
    for u in range(N):
        for x in range(N):
            temp_matrix[u] += input_matrix[x] * cos_matrix[x, u]

    # Matrice per il risultato finale della DCT
    output_matrix = np.zeros_like(input_matrix, dtype=float)

    # Applica la DCT alle colonne
    for v in range(N):
        for y in range(N):
            output_matrix[:, v] += temp_matrix[:, y] * cos_matrix[y, v]

    # Fattore di normalizzazione
    W = np.sqrt(2 / N) * np.ones(N)
    W[0] = np.sqrt(1 / N)

    # Applica i fattori di normalizzazione
    W = W[:, None] * W
    output_matrix *= W

    return output_matrix


if __name__ == '__main__':
    # Esempio di utilizzo
    matrix = np.array([[231, 32, 233, 161, 24, 71, 140, 245],
                             [247, 40, 248, 245, 124, 204, 36, 107],
                             [234, 202, 245, 167, 9, 217, 239, 173],
                             [193, 190, 100, 167, 43, 180, 8, 70],
                             [11, 24, 210, 177, 81, 243, 8, 112],
                             [97, 195, 203, 47, 125, 114, 165, 181],
                             [193, 70, 174, 167, 41, 30, 127, 245],
                             [87, 149, 57, 192, 65, 129, 178, 228]])

    dct_matrix = dct2_home_made(matrix)
    print(dct_matrix)
