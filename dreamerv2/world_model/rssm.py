import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td
from torch.distributions import OneHotCategoricalStraightThrough


class RSSM(nn.Module):
    def __init__(
        self,
        mlp_hidden_dim: int,
        rnn_hidden_dim: int,
        state_dim: int,
        num_classes: int,
        action_dim: int,
        obs_embed_dim: int,
    ):
        super().__init__()
        self.rnn_hidden_dim = rnn_hidden_dim
        self.state_dim = state_dim
        self.num_classes = num_classes

        self.transition_hidden = nn.Linear(state_dim * num_classes + action_dim, mlp_hidden_dim)
        self.transition = nn.GRUCell(mlp_hidden_dim, rnn_hidden_dim)

        self.prior_hidden = nn.Linear(rnn_hidden_dim, mlp_hidden_dim)
        self.prior_logits = nn.Linear(mlp_hidden_dim, state_dim * num_classes)

        self.posterior_hidden = nn.Linear(rnn_hidden_dim + obs_embed_dim, mlp_hidden_dim)
        self.posterior_logits = nn.Linear(mlp_hidden_dim, state_dim * num_classes)

    def recurrent(self, state: torch.Tensor, action: torch.Tensor, rnn_hidden: torch.Tensor):
        hidden = F.elu(self.transition_hidden(torch.cat([state, action], dim=1)))
        return self.transition(hidden, rnn_hidden)

    def get_prior(self, rnn_hidden: torch.Tensor, detach: bool = False):
        hidden = F.elu(self.prior_hidden(rnn_hidden))
        logits = self.prior_logits(hidden).reshape(-1, self.state_dim, self.num_classes)
        prior_dist = td.Independent(OneHotCategoricalStraightThrough(logits=logits), 1)
        if detach:
            detach_dist = td.Independent(OneHotCategoricalStraightThrough(logits=logits.detach()), 1)
            return prior_dist, detach_dist
        return prior_dist

    def get_posterior(self, rnn_hidden: torch.Tensor, embedded_obs: torch.Tensor, detach: bool = False):
        hidden = F.elu(self.posterior_hidden(torch.cat([rnn_hidden, embedded_obs], dim=1)))
        logits = self.posterior_logits(hidden).reshape(-1, self.state_dim, self.num_classes)
        posterior_dist = td.Independent(OneHotCategoricalStraightThrough(logits=logits), 1)
        if detach:
            detach_dist = td.Independent(OneHotCategoricalStraightThrough(logits=logits.detach()), 1)
            return posterior_dist, detach_dist
        return posterior_dist
