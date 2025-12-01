import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, joint_dim: int, joint_embed_dim: int = 128):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 48, kernel_size=4, stride=2)
        self.conv2 = nn.Conv2d(48, 96, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(96, 192, kernel_size=4, stride=2)
        self.conv4 = nn.Conv2d(192, 384, kernel_size=4, stride=2)
        self.joint_embed = nn.Sequential(
            nn.Linear(joint_dim, joint_embed_dim),
            nn.ELU(),
            nn.Linear(joint_embed_dim, joint_embed_dim),
            nn.ELU(),
        )
        self.output_dim = 1536 + joint_embed_dim

    def forward(self, images: torch.Tensor, joints: torch.Tensor):
        hidden = F.elu(self.conv1(images))
        hidden = F.elu(self.conv2(hidden))
        hidden = F.elu(self.conv3(hidden))
        image_embedding = self.conv4(hidden).reshape(images.size(0), -1)
        joint_embedding = self.joint_embed(joints)
        return torch.cat([image_embedding, joint_embedding], dim=1)
