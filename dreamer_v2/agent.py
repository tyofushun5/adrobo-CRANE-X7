import torch
import torch.nn as nn

from dreamer_v2.utils import preprocess_obs


class Agent(nn.Module):
    def __init__(self, encoder, decoder, rssm, action_model):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.rssm = rssm
        self.action_model = action_model
        self.device = next(self.action_model.parameters()).device
        self.rnn_hidden = torch.zeros(1, rssm.rnn_hidden_dim, device=self.device)

    def __call__(self, obs, eval: bool = True):
        obs = preprocess_obs(obs)
        image = torch.as_tensor(obs["image"], device=self.device)
        image = image.transpose(1, 2).transpose(0, 1).unsqueeze(0)
        joint = torch.as_tensor(obs["joint_pos"], device=self.device).unsqueeze(0)
        with torch.no_grad():
            state_prior = self.rssm.get_prior(self.rnn_hidden)
            state = state_prior.sample().flatten(1)
            obs_dist = self.decoder(state, self.rnn_hidden)
            obs_pred = obs_dist.mean

            embedded_obs = self.encoder(image, joint)
            state_posterior = self.rssm.get_posterior(self.rnn_hidden, embedded_obs)
            state = state_posterior.sample().flatten(1)
            action, _, _ = self.action_model(state, self.rnn_hidden, eval=eval)
            self.rnn_hidden = self.rssm.recurrent(state, action, self.rnn_hidden)

        reconstructed = (obs_pred.squeeze().cpu().numpy().transpose(1, 2, 0) + 0.5).clip(0.0, 1.0)
        return action.squeeze().cpu().numpy(), reconstructed

    def reset(self):
        self.rnn_hidden = torch.zeros_like(self.rnn_hidden)

    def to(self, device):
        # Match torch.nn.Module.to signature by returning self so callers can chain.
        self.device = device
        super().to(device)
        self.encoder.to(device)
        self.decoder.to(device)
        self.rssm.to(device)
        self.action_model.to(device)
        self.rnn_hidden = self.rnn_hidden.to(device)
        return self
