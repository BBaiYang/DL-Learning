import torch
from torch.utils import data
from LinearRegression.linear_regression_from_scratch import synthetic_data
from torch import nn


def load_array(data_arrays, batch_size, is_train=True):
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


if __name__ == '__main__':
    '''raw data'''
    true_w = torch.tensor([2, -3.4]).view(-1, 1)
    true_b = 4.2
    features, labels = synthetic_data(true_w, true_b)

    '''hyper-parameters'''
    num_epochs = 3
    lr = .03
    batch_size = 10

    data_iter = load_array((features, labels), batch_size)

    '''model'''
    net = nn.Sequential(nn.Linear(2, 1))

    '''initialized parameters'''
    net[0].weight.data.normal_(0, .01)
    net[0].bias.data.fill_(0)

    '''loss function'''
    loss = nn.MSELoss()

    '''optimizer'''
    trainer = torch.optim.SGD(net.parameters(), lr=lr)

    '''training process'''
    for epoch in range(num_epochs):
        for X, y in data_iter:
            l = loss(net(X), y)
            trainer.zero_grad()
            l.backward()
            trainer.step()
        train_l = loss(net(features), labels)
        print(f'epoch {epoch + 1}, loss {train_l:.5f}')

    w = net[0].weight.data
    b = net[0].bias.data
    print(f'estimated w: {w}, error in estimating w: {true_w - w.view(true_w.size())}')
    print(f'estimated b: {b}, error in estimating b: {true_b - b}')
