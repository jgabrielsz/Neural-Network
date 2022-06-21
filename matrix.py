import math
from random import gauss


class matrix:

    def __init__(self, rows, cols, randomize=False):
        self.rows = rows
        self.cols = cols

        self.matrix = list()
        if randomize:
            self.randomize()
        else:
            for _ in range(rows):
                row = list()
                for _ in range(cols):
                    row.append(0)
                self.matrix.append(row)

    @staticmethod
    def array_to_matrix(array) -> 'matrix':
        """
        Convert array to matrix

        This static method convert an
        array to a matrix with one column.

        Parameters:
        array: array to be converted

        Return:
        obj Matrix
        """
        m = matrix(len(array), 1)
        for i in range(len(array)):
            m.matrix[i][0] = array[i]
        return m

    @staticmethod
    def map(func, m1, m2) -> 'matrix':
        """
        Apply a function on both matrix rows.

        This static method apply a 'map' function on each row of the
        two matrix coleted.

        Parameters:
        func: the function
        m1: matrix 1
        m2: matrix 2

        Return:
        obj matrix
        """
        listmatrix = matrix(m1.rows, m1.cols)
        for row in range(len(m1.matrix)):
            listmatrix.matrix[row] = list(
                map(func, m1.matrix[row], m2.matrix[row]))
        return listmatrix

    @staticmethod
    def add(m1, m2) -> 'matrix':
        """
        sum two matrix

        This static method add two matrix

        Parameters:
        m1: matrix 1 
        m2: matrix 2

        Return:
        obj matrix
        """
        newMatrix = matrix.map(lambda x, y: x + y, m1, m2)
        return newMatrix

    @staticmethod
    def multiply(m1, m2) -> 'matrix':
        """
        Multiply two matrix

        This static method multiply two matrix

        Parameters:
        m1: matrix 1 
        m2: matrix 2

        Return:
        obj matrix
        """
        if m1.cols != m2.rows:
            print("Matrices can't be multiplied")
            return
        newMatrix = matrix(m1.rows, m2.cols)
        for row in range(len(m1.matrix)):
            for col in range(len(m2.matrix[0])):
                sum = 0
                for i in range(len(m1.matrix[0])):
                    sum += m1.matrix[row][i] * m2.matrix[i][col]
                newMatrix.matrix[row][col] = sum
        return newMatrix

    def randomize(self) -> None:
        """
        Randomize a matrix

        This method randomize all elements in a matrix

        Parameters: 
        none

        Return:
        none
        """
        for _ in range(self.rows):
            row = list()
            for _ in range(self.cols):
                row.append(gauss(0, 1))
            self.matrix.append(row)

    def sigmoid(self) -> None:
        """
        Aplly the sigmoid function

        This method aplly the sigmoid function
        on all the elemens in a matrix

        Parameters:
        None

        Return:
        None
        """
        def sgm(x): return 1 / (1 + math.exp(-x))

        for row in range(len(self.matrix)):
            for col in range(len(self.matrix[row])):
                self.matrix[row][col] = sgm(self.matrix[row][col])

    def show(self) -> None:
        """This method make the matrix visualization more friendly"""
        for row in self.matrix:
            print(row)
        print()

