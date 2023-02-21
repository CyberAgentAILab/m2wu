import numpy as np


class MWU(object):
    def __init__(self, eta, n_actions, decay=False):
        self.n_actions = n_actions
        self.logits = np.zeros(n_actions)
        self.strategy = np.ones(n_actions) / n_actions
        self.eta = eta
        self.sum_strategy = np.zeros(n_actions)
        self.decay = decay
        self.update_t = 0

    def update(self, utility):
        if self.decay:
            self.eta = 1 / (self.update_t + 1) ** (3 / 4)
        d_strategy = self.eta * utility
        self.logits += d_strategy
        exp_utility = np.exp(d_strategy) * self.strategy
        self.strategy = exp_utility / exp_utility.sum()
        self.update_t += 1
