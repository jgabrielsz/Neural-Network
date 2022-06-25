from posixpath import split
from matrix import *
from NeuralNetwork import *

NN = NeuralNetwork(2, 3, 1)

dataset = [# XOR problem
    [# inputs
       [1, 1], 
       [1, 0], 
       [0, 1], 
       [0, 0]
    ],
    [# outputs
        [0], 
        [1], 
        [1], 
        [0]
    ]
]

while True:
    for _ in range(20000):
        for i, element in enumerate(dataset[0]):
            NN.train(element, dataset[1][i])
    break


while True:
    i = input("Teste: ")
    inputs= i.split(",")

    for i, inp in enumerate(inputs):
       inputs[i] = int(inp)
    
    print(NN.predict(inputs))
