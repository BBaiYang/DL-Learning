from SoftmaxRegression.image_dataset import load_data_fashion_mnist
import torch.nn as nn
import torch
from SoftmaxRegression.softmax_regression_from_scratch import train_ch3


def relu(X):
    return torch.max(torch.zeros_like(X), X)


def net(X):
    X = X.reshape((-1, num_inputs))
    H = relu(X @ W1 + b1)
    return H @ W2 + b2


if __name__ == '__main__':
    '''hyper-parameters'''
    num_epochs = 10
    lr = .1
    batch_size = 256

    train_iter, test_iter = load_data_fashion_mnist(batch_size)

    '''model architecture'''
    num_inputs, num_outputs, num_hiddens = 784, 10, 256

    W1 = nn.Parameter(torch.randn(
        num_inputs, num_hiddens, requires_grad=True) * 0.01)
    b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
    W2 = nn.Parameter(torch.randn(
        num_hiddens, num_outputs, requires_grad=True) * 0.01)
    b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))

    params = [W1, b1, W2, b2]

    '''loss function'''
    loss = nn.CrossEntropyLoss()

    '''trainer'''
    updater = torch.optim.SGD(params, lr=lr)

    train_ch3(net, train_iter, test_iter, loss, num_epochs, updater, 'multilayer_perceptrons.png')
