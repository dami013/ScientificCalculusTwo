from DCT_lib import dct_rows_only, personal_dct2, dct2_library
import numpy as np

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

    # Test prima riga (DCT su righe)
    dct_matrix_rows = dct_rows_only(matrix)
    print("DCT applied to rows only:")
    print(dct_matrix_rows[0])

    # Applica la DCT2
    print("************DCT2************")
    print(personal_dct2(matrix))

    # Applica la DCT Scipy alla matrice
    print("************DCT2************")
    print(dct2_library(matrix))
