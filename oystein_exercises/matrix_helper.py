from numpy import mat, matrix, ndarray
from dataclasses import dataclass
import numpy as np


@dataclass
class Matrix:
    matrix: ndarray

    n: int = None
    P_L: ndarray = None
    P_U: ndarray = None
    P: ndarray = None

    def get_matrix(self):
        return matrix

    def get_element(self, row_no: int, col_no: int):
        return self.matrix[row_no - 1, col_no - 1]

    def set_matrix_size(self):
        self.n = int(np.sqrt(self.matrix.size))

    def LU_factorisation(self, input_matrix=None):
        """
        Takes a matrix as input and make a LU_factorisation of A
        and returing the matrices

        Page49 in book
        Page60 in pdf

        :return:
            lower and upper triangular matrices
        """

        # if input_matrix != None, then PLU-factorisation is running with PA
        if isinstance(input_matrix, ndarray):
            A = matrix_multiplication(input_matrix, self.matrix)
        else:
            A = self.matrix

        n = self.n

        L = np.zeros([n, n])
        U = np.zeros([n, n])

        # The first row of U: u_1j = a_1j, j = 1, 2, ..., n
        for j in range(1, n + 1):
            a_1j = get_element_in_matrix(matrix=A, row_no=1, col_no=j)

            set_element_in_matrix(matrix=U, row_no=1,
                                  col_no=j, element=a_1j)

        # The first column of L: l_11 = 1, l_i1 = a_i1/u_11, i = 2, 3, ..., n
        set_element_in_matrix(matrix=L, row_no=1, col_no=1, element=1)

        u_11 = get_element_in_matrix(matrix=A, row_no=1, col_no=1)

        for i in range(2, n + 1):
            a_i1 = get_element_in_matrix(matrix=A, row_no=i, col_no=1)

            set_element_in_matrix(matrix=L, row_no=i,
                                  col_no=1, element=a_i1/u_11)

        # L has only ones at the diagonal
        for i in range(1, n+1):
            set_element_in_matrix(matrix=L, row_no=i, col_no=i, element=1)

        """
        i = 2:
            l_ij, j = 1, ..., i - 1
            u_ij, j = i, ..., n
        then go over to i = 3
        """

        for i in range(2, n + 1):
            # Calculating l_ij
            for j in range(1, i):
                u_jj = get_element_in_matrix(matrix=U, row_no=j, col_no=j)
                a_ij = get_element_in_matrix(matrix=A, row_no=i, col_no=j)
                sum = 0
                for k in range(1, j):
                    l_ik = get_element_in_matrix(
                        matrix=L, row_no=i, col_no=k)
                    u_kj = get_element_in_matrix(
                        matrix=U, row_no=k, col_no=j)
                    sum += l_ik*u_kj

                # Replace the element l_ij
                l_ij = (1/u_jj) * (a_ij - sum)
                set_element_in_matrix(
                    matrix=L,
                    row_no=i,
                    col_no=j,
                    element=l_ij
                )

            # Calculating u_ij
            for j in range(i, n + 1):
                a_ij = get_element_in_matrix(matrix=A, row_no=i, col_no=j)
                sum = 0
                for k in range(1, i):
                    l_ik = get_element_in_matrix(
                        matrix=L, row_no=i, col_no=k)
                    u_kj = get_element_in_matrix(
                        matrix=U, row_no=k, col_no=j)
                    sum += l_ik*u_kj

                # Replace the element u_ij
                u_ij = a_ij - sum
                set_element_in_matrix(
                    matrix=U,
                    row_no=i,
                    col_no=j,
                    element=u_ij
                )

        # Rounding the elements inside the matrix
        L = np.around(L, decimals=1)
        U = np.around(U, decimals=1)
        return L, U

    def PLU_factorisation(self):

        n = matrix_get_size(self.matrix)

        permutation_list = []
        identity_matrix = np.identity(n)

        for k in range(n):
            col = matrix_get_col(self.matrix, k + 1)
            r = np.argmax(col) + 1

            switched = matrix_switch_to_rows(identity_matrix, k+1, r)
            permutation_list.append(switched)

        P = identity_matrix
        for P_element in permutation_list:
            P = matrix_multiplication(P_element, P)

        # Updating the P
        self.P_L, self.P_U = self.LU_factorisation(P)
        self.P = P

        return self.P_L, self.P_U, self.P, self.matrix

    def __str__(self):
        print(matrix)
        return f'matrix'


def matrix_switch_to_rows(matrix: ndarray, rad_no1: int, rad_no2: int):
    """
    A function for switching two rows in a matrix

    :param:

    :return:
        None
    """

    matrix[[rad_no1 - 1, rad_no2 - 1]] = matrix[[rad_no2 - 1, rad_no1 - 1]]
    return matrix


def matrix_get_size(matrix: ndarray):
    return int(np.sqrt(matrix.size))


def matrix_get_row(matrix: ndarray, row_no: int):
    return matrix[row_no - 1]


def matrix_get_col(matrix: ndarray, col_no: int):
    print(col_no)
    return matrix[:, col_no - 1]


def set_element_in_matrix(matrix: ndarray, row_no: int, col_no: int, element: int):
    matrix[row_no - 1, col_no - 1] = element


def get_element_in_matrix(matrix: ndarray, row_no: int, col_no: int):
    return matrix[row_no - 1, col_no - 1]


def matrix_multiplication(matrix1: ndarray, matrix2: ndarray):
    return np.matmul(matrix1, matrix2)


def matrix_inverse(matrix):
    return np.linalg.inv(matrix)


def testing_LU(A: ndarray, L: ndarray, U: ndarray, n: int, print_to_console: bool = False):
    LU = matrix_multiplication(L, U)
    passed = sum(sum(A == LU)) == n**2

    if print_to_console:
        print('----------------------------------')
        print(f'\nL Triangular')
        print(L)
        print(f'\nUpper Triangular ')
        print(U)
        print(f'\nA')
        print(A)
        print(f'\nLU')
        print(LU)
        print('----------------------------------')
    print(f'\nPassed the test: {passed}\n')


def testing_PLU(P: ndarray, A: ndarray, L: ndarray, U: ndarray, n: int, print_to_console: bool = False):
    PA = matrix_multiplication(P, A)
    LU = matrix_multiplication(L, U)
    passed = sum(sum(PA == LU)) == n**2

    if print_to_console:
        print('----------------------------------')
        print(f'\nPermutation')
        print(P)
        print(f'A')
        print(A)
        print(f'\nLower Triangular')
        print(L)
        print(f'\nUpper Triangular ')
        print(U)
        print(f'\nPA')
        print(PA)
        print(f'\nLU')
        print(LU)
        print('----------------------------------')
    print(f'\nPassed the test: {passed}\n')


if __name__ == '__main__':
    main_matrix = Matrix(np.array([[1, 1, 2], [1, 4, 3], [-1, 5, -4]]))
    main_matrix.set_matrix_size()
    # L, U = A.LU_factorisation()
    # testing_LU(A, L, U, True)

    L_P, U_P, P, A = main_matrix.PLU_factorisation()
    n = main_matrix.n

    print(f'Print out P:')
    print(P)

    testing_PLU(P, A, L_P, U_P, n, True)
