import numpy as np
import pandas as pd
import os

from collections import deque

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
from pandas_datareader import data as pdr

yf.pdr_override()


def download_data(tickers) -> None:
    # Download data from yahoo finance
    # store in the data folder
    if not os.path.isdir('data'):
        os.mkdir('data')
    
    yf_df = pdr.get_data_yahoo(tickers)
    for ticker in tickers:
        yf_df.index = yf_df.index.map(str)
        ticker_df = yf_df.T.loc[(slice(None), ticker),:].swaplevel(i=0,j=1).xs(ticker, level=0).T.dropna(how='all')
        ticker_df.to_csv('data/' + ticker + '.csv')

def get_column_scalers(ticker_df, selected_features) -> tuple[dict, pd.DataFrame]:
    column_scaler = {}
    for column in selected_features:
        scaler = MinMaxScaler()
        ticker_df[column] = scaler.fit_transform(np.expand_dims(ticker_df[column].values, axis=1))
        column_scaler[column] = scaler
    
    return column_scaler, ticker_df

def get_sequences(ticker_df, selected_features, label, n_steps=50, lookup_step=1) -> tuple[np.array, list]:
    last_sequence = np.array(ticker_df[selected_features].tail(lookup_step))
    ticker_df.dropna(inplace=True)
    sequence_data = []
    sequences = deque(maxlen=n_steps)

    for entry, target in zip(ticker_df[selected_features].values, ticker_df[label].values):
        sequences.append(entry)
        if len(sequences) == n_steps:
            sequence_data.append([np.array(sequences), target])
    
    last_sequence = list(sequences) + list(last_sequence)
    last_sequence = np.array(last_sequence)
    return last_sequence, sequence_data

def get_train_test_split(sequence_data, shuffle=False, test_size=0.2) -> list:
    X, y = [], []
    for seq, target in sequence_data:
        X.append(seq)
        y.append(target)
    X = np.array(X)
    y = np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], X.shape[2]))
    return train_test_split(X, y, test_size=test_size, shuffle=shuffle)

def get_data(ticker, selected_features=['Adj Close', 'Open', 'High', 'Low', 'Volume'], lookup_step=1) -> dict:
    # Load data from csv file
    ticker_df = pd.read_csv('data/' + ticker + '.csv')
    # Predicted following day's closing price
    label = 'Future'
    ticker_df.dropna(inplace=True)

    data = {}
    data['column_scaler'], ticker_df = get_column_scalers(ticker_df, selected_features)
    ticker_df[label] = ticker_df['Adj Close'].shift(-lookup_step)
    data['last_sequence'], sequence_data = get_sequences(ticker_df, selected_features, label)
    data['X_train'], data['X_test'], data['y_train'], data['y_test'] = get_train_test_split(sequence_data)
    return data

