import numpy as np

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
    N = input_matrix.shape[0]  # Numero di righe

    j = np.arange(N)  # Indici fino a N
    i = j[:, None]  # Indici fino a N-1

    # Calcola gli argomenti del coseno per ciascuna coppia di indici (i, j).
    cos_matrix = np.cos(np.pi * (2 * i + 1) * j / (2 * N))

    # Matrice per il risultato della DCT colonna per colonna
    output_matrix = np.zeros_like(input_matrix, dtype=float)

    # Applica la DCT alle colonne
    for v in range(input_matrix.shape[1]):  # Numero di colonne
        for y in range(N):
            output_matrix[y, v] = np.sum(input_matrix[:, v] * cos_matrix[:, y])

    # Fattore di normalizzazione
    W = np.sqrt(2 / N) * np.ones(N)
    W[0] = np.sqrt(1 / N)

    # Applica i fattori di normalizzazione
    output_matrix *= W[:, None]

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

    # Applica la DCT alle righe
    dct_matrix_rows = dct_rows_only(matrix)
    print("DCT applied to rows only:")
    print(dct_matrix_rows)

    # Applica la DCT alle colonne della matrice risultante
    dct_matrix_columns = dct_columns_only(dct_matrix_rows)
    print("DCT applied to columns (after rows):")
    print(dct_matrix_columns)
