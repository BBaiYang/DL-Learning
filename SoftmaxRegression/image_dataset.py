import torch
import torchvision
from torchvision import transforms
from torch.utils import data


def load_data_fashion_mnist(batch_size, num_workers=4, resize=None):
    """Download the Fashion-MNIST dataset and then load it into memory."""
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root='~/CODES/Dataset', train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(root='~/CODES/Dataset', train=False, transform=trans, download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=num_workers),
            data.DataLoader(mnist_test, batch_size, shuffle=False, num_workers=num_workers))


if __name__ == '__main__':
    train_iter, test_iter = load_data_fashion_mnist(32, resize=64)
    for X, y in train_iter:
        print(X.shape, X.dtype, y.shape, y.dtype)
        break
