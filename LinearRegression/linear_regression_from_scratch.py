import torch
import random


def synthetic_data(w, b, num_examples=1000):
    X = torch.normal(0, 1, (num_examples, w.size(0)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, .01, y.size())
    return X, y


def data_iter(batch_size, features, labels):
    num_examples = features.size(0)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]


def model(X, w, b):
    return torch.matmul(X, w) + b


def squared_loss(y_hat, y):
    return (y_hat - y.view(y_hat.size())) ** 2 / 2


def sgd(params, lr, batch_size):
    for param in params:
        param.data.sub_(lr * param.grad / batch_size)
        param.grad.data.zero_()


if __name__ == '__main__':
    '''raw data'''
    true_w = torch.tensor([2, -3.4]).view(-1, 1)
    true_b = 4.2
    features, labels = synthetic_data(true_w, true_b)

    '''random initialized model'''
    w = torch.normal(0, .01, size=(2, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)

    '''hyper-parameters'''
    num_epochs = 3
    lr = .03
    batch_size = 10
    net = model
    loss = squared_loss

    '''training process'''
    for epoch in range(num_epochs):
        for X, y in data_iter(batch_size, features, labels):
            l = loss(net(X, w, b), y)
            l.sum().backward()
            sgd([w, b], lr, batch_size)
        with torch.no_grad():
            train_l = loss(net(features, w, b), labels)
            print(f'epoch {epoch + 1}, loss {float(train_l.mean()):.5f}')

    print(f'estimated w: {w}, error in estimating w: {true_w - w.view(true_w.size())}')
    print(f'estimated b: {b}, error in estimating b: {true_b - b}')
