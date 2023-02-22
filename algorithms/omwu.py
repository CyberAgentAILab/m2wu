import numpy as np

from algorithms import MWU


class OMWU(MWU):
    def __init__(self, n_actions, random_init_strategy, eta, decay=False):
        super().__init__(n_actions, random_init_strategy, eta, decay)
        self.past_utility = np.zeros(n_actions)

    def update(self, utility):
        if self.decay:
            self.eta = 1 / (self.t + 1) ** (3 / 4)
        values = np.exp(self.eta * (2 * utility - self.past_utility)) * self.strategy
        self.strategy = values / values.sum()
        self.t += 1
        self.past_utility = utility
