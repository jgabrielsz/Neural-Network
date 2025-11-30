from matrix import *
from NeuralNetwork import *
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
import pandas as pd


def normalize_dataset(df: pd.DataFrame, target: str = "quality"):
    X = df.drop(target, axis=1)
    y = df[[target]]

    scaler_X = MinMaxScaler()
    X_normalized = scaler_X.fit_transform(X)

    scaler_y = MinMaxScaler()
    y_normalized = scaler_y.fit_transform(y)

    df_normalized = pd.DataFrame(X_normalized, columns=X.columns)
    df_normalized[target] = y_normalized

    return df_normalized, scaler_X, scaler_y


def convert_dataset(df: pd.DataFrame):
    input_columns = df.columns[:-1]
    output_column = df.columns[-1]

    inputs = df[input_columns].values.tolist()
    outputs = df[[output_column]].values.tolist()

    return [inputs, outputs]


def divide_dataset(df: pd.DataFrame, train_ratio=0.4, val_ratio=0.3, test_ratio=0.3):
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio deve ser igual a 1.0")

    shuffled = shuffle(df, random_state=None)
    n = len(shuffled)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train_df = shuffled.iloc[:train_end]
    val_df = shuffled.iloc[train_end:val_end]
    test_df = shuffled.iloc[val_end:]

    return convert_dataset(train_df), convert_dataset(val_df), convert_dataset(test_df)
