import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td
from torch.distributions import OneHotCategoricalStraightThrough


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 48, kernel_size=4, stride=2)
        self.conv2 = nn.Conv2d(48, 96, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(96, 192, kernel_size=4, stride=2)
        self.conv4 = nn.Conv2d(192, 384, kernel_size=4, stride=2)

    def forward(self, obs: torch.Tensor):
        """
        観測画像をベクトルに埋め込むためのEncoder．

        Parameters
        ----------
        obs : torch.Tensor (B, C, H, W)
            入力となる観測画像．

        Returns
        -------
        embedded_obs : torch.Tensor (B, D)
            観測画像をベクトルに変換したもの．Dは入力画像の幅と高さに依存して変わる．
            入力が(B, 3, 64, 64)の場合，出力は(B, 1536)になる．
        """
        hidden = F.elu(self.conv1(obs))
        hidden = F.elu(self.conv2(hidden))
        hidden = F.elu(self.conv3(hidden))
        embedded_obs = self.conv4(hidden).reshape(hidden.size(0), -1)

        return embedded_obs  # x_t