import torch
import math
import numpy as np
import time
import matplotlib.pyplot as plt


class Timer:
    """Record multiple running times."""
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """Start the timer."""
        self.tik = time.time()

    def stop(self):
        """Stop the timer and record the time in a list."""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """Return the average time."""
        return sum(self.times) / len(self.times)

    def sum(self):
        """Return the sum of time."""
        return sum(self.times)

    def cumsum(self):
        """Return the accumulated time."""
        return np.array(self.times).cumsum().tolist()


def normal(x, mu, sigma):
    return 1 / np.sqrt(2 * math.pi * sigma ** 2) * np.exp(- ((x - mu) ** 2) / (2 * sigma ** 2))


if __name__ == '__main__':
    n = 10000
    a = torch.ones(n)
    b = torch.ones(n)
    c = torch.empty(n)

    timer = Timer()
    for i in range(n):
        c[i] = a[i] + b[i]
    print('%.5f sec' % timer.stop())

    timer.start()
    d = a + b
    print('%.5f sec' % timer.stop())

    x = np.arange(-7, 7, .01)
    params = [(0, 1), (0, 2), (3, 1)]
    for mu, sigma in params:
        plt.plot(x, normal(x, mu, sigma), label=f'mean {mu}, std {sigma}')
    plt.legend()
    plt.savefig('normal.png')
