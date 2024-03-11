import numpy as np
import torch


class NullModel:
    def __init__(self):
        self.training = None
        self.device = None

    def train(self, mode=None):
        pass

    def to(self, device=None):
        pass

    @staticmethod
    def parameters() -> None:
        return None


class RandomPolicy(NullModel):
    def __init__(self):
        super().__init__()

    @staticmethod
    def select_action(x, mean_action=True):
        edge_mask = x[0][-2]
        action = torch.zeros(1)
        valid_actions = torch.nonzero(edge_mask.flatten()).flatten()

        if len(valid_actions) > 0:
            index = torch.randint(0, len(valid_actions), (1,1))
            action[0] = valid_actions[index]

        return action

class GreedyPolicy(NullModel):
    def __init__(self):
        super().__init__()

    @staticmethod
    def select_action(x, mean_action=True):
        edge_mask = x[0][-2]
        action = torch.zeros(1)
        valid_actions = torch.nonzero(edge_mask.flatten()==2).flatten()

        if len(valid_actions) > 0:
            index = torch.randint(0, len(valid_actions), (1,1))
            action[0] = valid_actions[index]
        else:
            valid_actions = torch.nonzero(edge_mask.flatten()).flatten()
            index = torch.randint(0, len(valid_actions), (1,1))
            action[0] = index

        return action