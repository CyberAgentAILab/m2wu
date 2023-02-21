import numpy as np

from algorithms import MWU


class OMWU(MWU):
    def __init__(self, eta, n_actions, decay=False):
        super().__init__(eta, n_actions, decay)
        self.past_utility = np.zeros(n_actions)

    def update(self, utility):
        if self.decay:
            self.eta = 1 / (self.update_t + 1) ** (3 / 4)
        d_strategy = self.eta * (2 * utility - self.past_utility)
        self.logits += d_strategy
        exp_utility = np.exp(d_strategy) * self.strategy
        self.strategy = exp_utility / exp_utility.sum()
        self.update_t += 1
        self.past_utility = utility
