import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td

from dreamer_v2.distributions import TruncNormalDist


class Actor(nn.Module):
    def __init__(self, action_dim: int, hidden_dim: int, rnn_hidden_dim: int, state_dim: int, num_classes: int):
        super().__init__()
        input_dim = state_dim * num_classes + rnn_hidden_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.std = nn.Linear(hidden_dim, action_dim)
        self.min_stddev = 0.1
        self.init_stddev = np.log(np.exp(5.0) - 1)

    def forward(self, state: torch.Tensor, rnn_hidden: torch.Tensor, eval: bool = False):
        hidden = F.elu(self.fc1(torch.cat([state, rnn_hidden], dim=1)))
        hidden = F.elu(self.fc2(hidden))
        hidden = F.elu(self.fc3(hidden))
        hidden = F.elu(self.fc4(hidden))
        mean = torch.tanh(self.mean(hidden))
        stddev = 2 * torch.sigmoid((self.std(hidden) + self.init_stddev) / 2) + self.min_stddev
        if eval:
            return mean, None, None
        dist = td.Independent(TruncNormalDist(mean, stddev, -1, 1), 1)
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy()
