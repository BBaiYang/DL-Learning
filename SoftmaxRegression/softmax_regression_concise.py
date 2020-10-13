import torch
from torch import nn
from SoftmaxRegression.image_dataset import load_data_fashion_mnist
from SoftmaxRegression.softmax_regression_from_scratch import train_ch3


class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


def init_weights(model):
    if type(model) == nn.Linear:
        nn.init.normal_(model.weight, std=.01)


if __name__ == '__main__':
    '''hyper-parameters'''
    num_epochs = 10
    lr = .1
    batch_size = 256

    train_iter, test_iter = load_data_fashion_mnist(batch_size)

    '''net architecture'''
    net = nn.Sequential(FlattenLayer(), nn.Linear(784, 10))
    net.apply(init_weights)

    '''loss'''
    loss = nn.CrossEntropyLoss()

    '''optimizer'''
    trainer = torch.optim.SGD(net.parameters(), lr=lr)
    train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
