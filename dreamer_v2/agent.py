import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td
from torch.distributions import OneHotCategoricalStraightThrough


class Agent(nn.Module):
    """
    ActionModelに基づき行動を決定する. そのためにRSSMを用いて状態表現をリアルタイムで推論して維持するクラス
    """
    def __init__(self, encoder, decoder, rssm, action_model):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.rssm = rssm
        self.action_model = action_model

        self.device = next(self.action_model.parameters()).device
        self.rnn_hidden = torch.zeros(1, rssm.rnn_hidden_dim, device=self.device)

    def __call__(self, obs, eval=True):
        # preprocessを適用, PyTorchのためにChannel-Firstに変換
        obs = preprocess_obs(obs)
        obs = torch.as_tensor(obs, device=self.device)
        obs = obs.transpose(1, 2).transpose(0, 1).unsqueeze(0)

        with torch.no_grad():
            # 現在の状態から次に得られる観測画像を予測する
            state_prior = self.rssm.get_prior(self.rnn_hidden)
            state = state_prior.sample().flatten(1)
            obs_dist = self.decoder(state, self.rnn_hidden)
            obs_pred = obs_dist.mean

            # 観測を低次元の表現に変換し, posteriorからのサンプルをActionModelに入力して行動を決定する
            embedded_obs = self.encoder(obs)
            state_posterior = self.rssm.get_posterior(self.rnn_hidden, embedded_obs)
            state = state_posterior.sample().flatten(1)
            action, _, _  = self.action_model(state, self.rnn_hidden, eval=eval)

            # 次のステップのためにRNNの隠れ状態を更新しておく
            self.rnn_hidden = self.rssm.recurrent(state, action, self.rnn_hidden)

        return action.squeeze().cpu().numpy(), (obs_pred.squeeze().cpu().numpy().transpose(1, 2, 0) + 0.5).clip(0.0, 1.0)

    #RNNの隠れ状態をリセット
    def reset(self):
        self.rnn_hidden = torch.zeros(1, self.rssm.rnn_hidden_dim, device=self.device)

    def to(self, device):
        self.device = device
        self.encoder.to(device)
        self.decoder.to(device)
        self.rssm.to(device)
        self.action_model.to(device)
        self.rnn_hidden = self.rnn_hidden.to(device)