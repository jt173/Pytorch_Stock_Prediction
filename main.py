import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import lr_scheduler

from data_loader import download_data, get_data
from lstm_model import LSTM

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

def main():
    # Params
    batch_size = 64
    max_epochs = 25

    #download_data(['AAPL'])
    data = get_data('AAPL')
    X_train = data['X_train']
    y_train = data['y_train']
    X_test = data['X_test']
    y_test = data['y_test']

    X_train = torch.Tensor(X_train)
    y_train = torch.Tensor(y_train)

    X_test = torch.Tensor(X_test)
    y_test = torch.Tensor(y_test)

    dataset_train = TensorDataset(X_train, y_train)
    train_loader = DataLoader(dataset_train, batch_size=batch_size)

    dataset_test = TensorDataset(X_test, y_test)
    test_loader = DataLoader(dataset_test, batch_size=batch_size)

    model = LSTM().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=1.0)

    criterion = nn.MSELoss()

    # Train
    for i in range(max_epochs):
        model.train()
        for seq, label in train_loader:
            seq = seq.to(device)
            label = label.view(-1, 1).to(device)
            # Set optimizer gradient to zero
            optimizer.zero_grad()
            # compute prediction
            y_pred = model(seq)
            loss = criterion(y_pred, label)
            # Backpropagation
            loss.backward()
            # Gradient descent step
            optimizer.step()

        scheduler.step()
        
        #torch.save(optimizer.state_dict(), os.path.join('results', 'optimizer') + '.pth')
        #torch.save(model.state_dict(), os.path.join('results', 'model') + '.pth')
        print(f'Epoch [{i + 1}/{max_epochs}] Loss: {loss.item():10.8f}')

    # Test
    model.eval()
    y_actual = data['y_test']
    y_actual = np.squeeze(data['column_scaler']['Adj Close'].inverse_transform(np.expand_dims(y_actual, axis=0)))

    y_pred = np.empty(1)

    # Predictions
    for seq, label in test_loader:
        optimizer.zero_grad()
        seq = seq.to(device)
        output = model(seq)
        N_STEPS = 50
        last_sequence = data['last_sequence'][-N_STEPS:]
        column_scaler = data['column_scaler']
        last_sequence = last_sequence.reshape((last_sequence.shape[1], last_sequence.shape[0]))
        # Expand dim
        last_sequence = np.expand_dims(last_sequence, axis=0)
        # Get the price by inverting the scaling
        predicted_price = column_scaler['Adj Close'].inverse_transform(output.cpu().detach().numpy())
        # Concatenate the predicted price to y_pred
        y_pred = np.concatenate((y_pred, predicted_price.squeeze()), axis=0)

    # Plot
    plt.plot(y_actual[1:], c='b')
    plt.plot(y_pred[1:], c='r')
    plt.xlabel('Days')
    plt.ylabel('Price')
    plt.legend(['Actual Price', 'Predicted Price'])
    plt.show()
        
if __name__ == '__main__':
    main()
