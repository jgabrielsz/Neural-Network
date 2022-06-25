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

    def show(self) -> None:
        for row in self.matrix:
            print(row)
        print()

    @staticmethod
    def map(func, m1, m2) -> 'matrix':
        
        listmatrix = matrix(m1.rows, m1.cols)
        for row in range(len(m1.matrix)):
            listmatrix.matrix[row] = list(
                map(func, m1.matrix[row], m2.matrix[row]))
        return listmatrix
    
    @staticmethod
    def add(m1, m2) -> 'matrix':
        newMatrix = matrix.map(lambda x, y: x + y, m1, m2)
        return newMatrix
    
    @staticmethod
    def subtract(m1, m2) -> 'matrix':     
        newMatrix = matrix.map(lambda x, y: x - y, m1, m2)
        return newMatrix

    @staticmethod
    def hadamard(m1, m2) -> 'matrix':
        newMatrix = matrix.map(lambda x, y: x * y, m1, m2)
        return newMatrix

    @staticmethod
    def array_to_matrix(array) -> 'matrix':
        m = matrix(len(array), 1)
        for i in range(len(array)):
            m.matrix[i][0] = array[i]
        return m

    def randomize(self) -> None:
        for _ in range(self.rows):
            row = list()
            for _ in range(self.cols):
                row.append(gauss(0, 1))
                #random positive number between 1 and 10
                #row.append(random.randint(1, 10))

            self.matrix.append(row)

    @staticmethod
    def multiply_escalar(m1, escalar) -> 'matrix':
        newMatrix = matrix(m1.rows, m1.cols)
        
        for row in range(len(m1.matrix)):
            for col in range(len(m1.matrix[row])):
                newMatrix.matrix[row][col] = m1.matrix[row][col] * escalar
            
        return newMatrix

    @staticmethod
    def transpose(matrix1) -> 'matrix':
        newMatrix = matrix(matrix1.cols, matrix1.rows)

        for row in range(len(matrix1.matrix)):
            for col in range(len(matrix1.matrix[row])):
                newMatrix.matrix[col][row] = matrix1.matrix[row][col]
        return newMatrix 

    def sigmoid(self) -> None:
        def sgm(x): return 1 / (1 + math.exp(-x))

        for row in range(len(self.matrix)):
            for col in range(len(self.matrix[row])):
                self.matrix[row][col] = sgm(self.matrix[row][col])
    
    @staticmethod
    def d_sigmoid(m1) -> None:
        for row in range(len(m1.matrix)):
            for col in range(len(m1.matrix[row])):
                m1.matrix[row][col] = m1.matrix[row][col] * (1-m1.matrix[row][col])
        return m1

    @staticmethod
    def multiply(m1, m2) -> 'matrix':
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

