import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td


class Actor(nn.Module):
    def __init__(self, action_dim: int, hidden_dim: int, rnn_hidden_dim: int, state_dim: int, num_classes: int):
        super().__init__()
        input_dim = state_dim * num_classes + rnn_hidden_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.logits = nn.Linear(hidden_dim, action_dim)
        self.action_dim = action_dim

    def forward(self, state: torch.Tensor, rnn_hidden: torch.Tensor, eval: bool = False):
        hidden = F.elu(self.fc1(torch.cat([state, rnn_hidden], dim=1)))
        hidden = F.elu(self.fc2(hidden))
        hidden = F.elu(self.fc3(hidden))
        hidden = F.elu(self.fc4(hidden))
        logits = self.logits(hidden)
        if eval:
            action_idx = torch.argmax(logits, dim=1)
            action = F.one_hot(action_idx, num_classes=self.action_dim).float()
            return action, None, None
        dist = td.OneHotCategoricalStraightThrough(logits=logits)
        action = dist.rsample()
        return action, dist.log_prob(action), dist.entropy()
