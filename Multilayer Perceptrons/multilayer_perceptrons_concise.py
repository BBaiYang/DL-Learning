from SoftmaxRegression.image_dataset import load_data_fashion_mnist
import torch.nn as nn
import torch
from SoftmaxRegression.softmax_regression_from_scratch import train_ch3


class MultilayerPerceptrons:
    def __init__(self):
        self.net = self._make_layers()
        self.net.apply(self.init_weights)

    def _make_layers(self):
        layers = []
        layers += [nn.Flatten()]
        layers += [nn.Linear(num_inputs, num_hiddens)]
        layers += [nn.ReLU()]
        layers += [nn.Linear(num_hiddens, num_outputs)]
        return nn.Sequential(*layers)

    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.normal_(m.weight, std=0.01)


if __name__ == '__main__':
    '''hyper-parameters'''
    num_epochs = 10
    lr = .1
    batch_size = 256

    train_iter, test_iter = load_data_fashion_mnist(batch_size)

    '''model architecture'''
    num_inputs, num_outputs, num_hiddens = 784, 10, 256
    mlp = MultilayerPerceptrons()

    '''loss'''
    loss = nn.CrossEntropyLoss()

    '''trainer'''
    updater = torch.optim.SGD(mlp.net.parameters(), lr=lr)

    train_ch3(mlp.net, train_iter, test_iter, loss, num_epochs, updater, 'multilayer_perceptrons.png')
