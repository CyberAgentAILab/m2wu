import numpy as np


class MWU(object):
    def __init__(self, n_actions, random_init_strategy, eta, decay=False):
        if random_init_strategy:
            self.strategy = np.random.exponential(scale=1.0, size=n_actions)
            self.strategy /= self.strategy.sum()
        else:
            self.strategy = np.ones(n_actions) / n_actions
        self.eta = eta
        self.decay = decay
        self.t = 0

    def update(self, utility):
        if self.decay:
            self.eta = 1 / (self.t + 1) ** (3 / 4)
        values = np.exp(self.eta * utility) * self.strategy
        self.strategy = values / values.sum()
        self.t += 1

    @classmethod
    def name(cls, params):
        return cls.__name__
