import torch
import matplotlib.pyplot as plt
from SoftmaxRegression.image_dataset import load_data_fashion_mnist
from LinearRegression.linear_regression_from_scratch import sgd


class Accumulator:
    """For accumulating sums over `n` variables."""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class Animator:
    def __init__(self, x_label=None, y_label=None, x_lim=None, y_lim=None, legends=None, fmts=None):
        assert len(legends) == len(fmts)
        self.x_label = x_label
        self.y_label = y_label
        self.x_lim = x_lim
        self.y_lim = y_lim
        self.legends = legends
        self.fmts = fmts
        self.x_axis = []
        self.y_axes = [[] for _ in range(len(legends))]

    def add(self, x, ys):
        self.x_axis.append(x)
        for i, y in enumerate(ys):
            self.y_axes[i].append(y)

    def display(self, fname):
        plt.figure(figsize=(8, 6))
        for y_axis, legend, fmt in zip(self.y_axes, self.legends, self.fmts):
            plt.plot(self.x_axis, y_axis, fmt, label=legend)
        plt.xlabel(self.x_label)
        plt.ylabel(self.y_label)
        plt.legend(loc='best')
        plt.savefig(fname)


def softmax(O):
    O_exp = torch.exp(O)
    return O_exp / O_exp.sum(dim=1, keepdim=True)


def model(X):
    return softmax(torch.matmul(X.view(-1, W.size(0)), W) + b)


def cross_entropy(y_hat, y):
    return -torch.log(y_hat[range(y_hat.size(0)), y])


def updater(batch_examples):
    return sgd([W, b], lr, batch_examples)


def accuracy(y_hat, y):
    cmp = (y_hat.argmax(dim=1).type(y.dtype) == y).sum().float()
    return cmp


def evaluate_accuracy(net, data_iter):
    if isinstance(net, torch.nn.Module):
        net.eval()
    metric = Accumulator(2)
    for X, y in data_iter:
        metric.add(accuracy(net(X), y), len(y))
    return metric[0] / metric[1]


def train_epoch_ch3(net, train_iter, loss, updater):
    if isinstance(net, torch.nn.Module):
        net.train()
    metric = Accumulator(3)
    for X, y in train_iter:
        batch_examples = X.size(0)
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            updater.step()
            metric.add(float(l) * batch_examples, accuracy(y_hat, y), batch_examples)
        else:
            l.sum().backward()
            updater(batch_examples)
            metric.add(float(l.sum()), accuracy(y_hat, y), batch_examples)
    return metric[0] / metric[2], metric[1] / metric[2]


def train_ch3(net, train_iter, test_iter, loss, num_epochs=10, updater=None, fname=None):
    animator = Animator('global epoch', 'accuracy', legends=('train accuracy', 'test accuracy'), fmts=('r-', 'b-'))
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, (train_acc, test_acc))
        print(f'epoch {epoch + 1}, '
              f'training loss {train_loss: .3f}, training accuracy {train_acc: .3f}, test accuracy {test_acc: .3f}')
    animator.display(fname)


if __name__ == '__main__':
    '''hyper-parameters'''
    num_epochs = 10
    lr = .1
    batch_size = 256

    train_iter, test_iter = load_data_fashion_mnist(batch_size)

    '''feature dimensionality and classes'''
    num_inputs = 784
    num_outputs = 10

    '''initialized parameters'''
    W = torch.normal(0, .01, size=(num_inputs, num_outputs), requires_grad=True)
    b = torch.zeros(num_outputs, requires_grad=True)

    '''training process'''
    train_ch3(model, train_iter, test_iter, cross_entropy, num_epochs, updater, 'softmax_regression.png')
