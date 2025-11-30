from matrix import *
from NeuralNetwork import *
import matplotlib.pyplot as plt
import pandas as pd
from preprocessing import *


NN = NeuralNetwork(11, 15, 1)
print("Treinando Rede Neural...")


def train_NN(NN, x_train, y_train, x_val, y_val, epochs=10000, early_stopping=True, patience=20):
    train_errors = []
    val_errors = []

    best_val_error = float("inf")
    epochs_without_improvement = 0

    for epoch in range(epochs):
        epoch_train_error = 0
        for i in range(len(x_train)):
            epoch_train_error += NN.train(x_train[i], y_train[i])
        epoch_train_error /= len(x_train)
        train_errors.append(epoch_train_error)

        epoch_val_error = 0
        for i in range(len(x_val)):
            NN.feedforward(x_val[i])
            epoch_val_error += NN.compute_error(y_val[i])
        epoch_val_error /= len(x_val)
        val_errors.append(epoch_val_error)

        print(f"Epoch {epoch+1}/{epochs}  |  Train Error: {epoch_train_error:.6f}  |  Val Error: {epoch_val_error:.6f}")

        if early_stopping and epoch >= 400:
            if epoch_val_error < best_val_error:
                best_val_error = epoch_val_error
                epochs_without_improvement = 0
                print(f"  -> New best val error: {best_val_error:.6f} (reset patience)")
            else:
                epochs_without_improvement += 1
                if epoch % 50 == 0:
                    print(f"  No improvement. epochs_without_improvement = {epochs_without_improvement}")

            if epochs_without_improvement >= patience:
                print("\nEarly stopping triggered!")
                break

    return train_errors, val_errors

df = pd.read_csv("winequality-red.csv")
df_normalized, scaler_X, scaler_y = normalize_dataset(df)

train, val, test = divide_dataset(df_normalized)

train_errors, val_errors = train_NN(
    NN,
    train[0], train[1],
    val[0], val[1],
    epochs=5000,
    early_stopping=True,
    patience=15
)

test_errors = []
y_pred_normalized = []

for i in range(len(test[0])):
    NN.feedforward(test[0][i])
    e = NN.compute_error(test[1][i])
    test_errors.append(e)
    
    y_pred_normalized.append([NN.output.matrix[0][0]])

mean_test_error = sum(test_errors) / len(test_errors)

import numpy as np
y_pred_real = scaler_y.inverse_transform(np.array(y_pred_normalized))
y_test_real = scaler_y.inverse_transform(np.array(test[1]))

rmse_real = np.sqrt(np.mean((y_test_real - y_pred_real)**2))
print(f"Mean Test Error (normalized scale): {mean_test_error:.6f}")
print(f"RMSE (real scale): {rmse_real:.4f}")

#Plotting errors
plt.plot(train_errors, label="Training Error")
plt.plot(val_errors, label="Validation Error")
plt.hlines(mean_test_error, xmin=0, xmax=len(train_errors), linestyles="dashed", label="Test Mean Error")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.show()
