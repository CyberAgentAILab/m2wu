import numpy as np


class MatrixGame(object):
    def __init__(self, payoff, guassian_scale = 0.1):
        self.payoff = payoff
        self.guassian_scale = guassian_scale

    def full_feedback(self, strategies):
        return [self.payoff @ strategies[1], -self.payoff.T @ strategies[0]]

    def noisy_feedback(self, strategies):
        loss = [self.payoff @ strategies[1], -self.payoff.T @ strategies[0]]
        loss[0] += np.random.normal(0, self.guassian_scale, self.payoff.shape[0]) 
        loss[1] += np.random.normal(0, self.guassian_scale, self.payoff.shape[1]) 
        return loss

    def calc_exploitability(self, strategies):
        return max(self.payoff @ strategies[1]) + max(-self.payoff.T @ strategies[0])

    def n_actions_tuple(self):
        return self.payoff.shape[0], self.payoff.shape[1]